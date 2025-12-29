import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

# PyTorch bellek fragmantasyonunu önler
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_sample_from_npy(npy_path: str) -> dict:
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
        sample = data.item()
    elif isinstance(data, dict):
        sample = data
    else:
        sample = data.item()

    if "rgb" not in sample:
        raise KeyError(f"'rgb' key not found in {npy_path}")
    return sample


def compute_exclusion_from_background(sample, bg_sample, user_threshold=10.0):
    depth_key = "depth_raw" if "depth_raw" in sample else "depth"
    has_depth = depth_key in sample and depth_key in bg_sample

    if has_depth:
        print(f"Using Depth ({depth_key}) for Background Subtraction...")
        curr = sample[depth_key]
        bg = bg_sample[depth_key]
        curr = np.nan_to_num(curr, nan=0.0, posinf=0.0, neginf=0.0)
        bg = np.nan_to_num(bg, nan=0.0, posinf=0.0, neginf=0.0)

        diff = cv2.absdiff(curr, bg)

        if diff.dtype == np.float32 or diff.dtype == np.float64:
            if user_threshold > 1.0:
                actual_threshold = 0.02
                print(f"   [INFO] Veri float. Threshold: {actual_threshold}m")
            else:
                actual_threshold = user_threshold
            _, fg_mask = cv2.threshold(diff, actual_threshold, 255, cv2.THRESH_BINARY)
        else:
            actual_threshold = int(user_threshold)
            print(f"   [INFO] Veri int. Threshold: {actual_threshold}")
            _, fg_mask = cv2.threshold(diff, actual_threshold, 255, cv2.THRESH_BINARY)
        fg_mask = fg_mask.astype(np.uint8)
    else:
        print("Depth not found, falling back to RGB...")
        curr = sample["rgb"]
        bg = bg_sample["rgb"]
        diff = cv2.absdiff(curr, bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, fg_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    exclusion_mask = fg_mask == 0
    return exclusion_mask, fg_mask


def render_segmentation_map(masks, H, W, background=0, seed=0):
    seg_vis = np.full((H, W, 3), background, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    masks_sorted = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
    for m in masks_sorted:
        color = rng.integers(50, 256, size=(3,), dtype=np.uint8)
        seg_vis[m["segmentation"]] = color
    return seg_vis


def exclusion_overlap_ratio(seg, excl):
    area = int(seg.sum())
    if area == 0:
        return 0.0
    overlap = int(np.logical_and(seg, excl).sum())
    return overlap / area


def filter_by_exclusion(masks, exclusion_mask, overlap_thresh=0.20):
    kept = []
    for m in masks:
        seg = m["segmentation"].astype(bool)
        r = exclusion_overlap_ratio(seg, exclusion_mask)
        if r < overlap_thresh:
            kept.append(m)
    return kept


# --- YENİ EKLENEN FONKSİYON: PARÇALARI BİRLEŞTİRME (MERGE) ---
def smart_merge_masks(masks, iou_thresh=0.1, ioa_thresh=0.6):
    """
    Parçalanmış maskeleri birleştirir.

    Args:
        iou_thresh: İki maske ne kadar kesişiyorsa birleşsin? (Düşük = az temas yeterli)
        ioa_thresh (Intersection over Area): Küçük maskenin ne kadarı büyük maskenin içinde?
                                           Eğer %60'ı içindeyse, onu parçası sayıp birleştirir.
    """
    if not masks:
        return []

    # Maskeleri alana göre sırala (Büyükten küçüğe)
    # Büyükleri "ana gövde" kabul edip küçükleri içine yutacağız.
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    merged_indices = set()
    final_masks = []

    for i in range(len(sorted_masks)):
        if i in merged_indices:
            continue

        # Mevcut maskeyi "Base" (Temel) olarak al
        base_mask = sorted_masks[i]["segmentation"].copy()
        base_area = sorted_masks[i]["area"]

        # Bu maskeye eklenecek diğer parçaları bul
        merged_any = False

        for j in range(i + 1, len(sorted_masks)):
            if j in merged_indices:
                continue

            cand_mask = sorted_masks[j]["segmentation"]
            cand_area = sorted_masks[j]["area"]

            # Kesişim (Intersection)
            intersection = np.logical_and(base_mask, cand_mask).sum()

            if intersection == 0:
                # Hiç değmiyorlarsa, belki çok yakındırlar? (Dilate ile kontrol)
                # İsteğe bağlı: Maskeleri biraz şişirip bakabiliriz ama şimdilik gerek yok.
                continue

            # Metrik 1: IoU (Klasik benzerlik)
            union = np.logical_or(base_mask, cand_mask).sum()
            iou = intersection / union

            # Metrik 2: IoMin (Küçük olanın ne kadarı büyük olanın içinde?)
            # Bu, "etiket kutunun üstünde" veya "kapak şişenin üstünde" durumunu yakalar.
            io_min = intersection / min(base_area, cand_area)

            # BİRLEŞTİRME KARARI:
            # 1. IoU yüksekse (zaten aynı yerdeler)
            # 2. VEYA Küçük parça büyük oranda diğerinin içindeyse (IoMin > ioa_thresh)
            if iou > iou_thresh or io_min > ioa_thresh:
                # Maskeleri birleştir (OR işlemi)
                base_mask = np.logical_or(base_mask, cand_mask)
                base_area = base_mask.sum()  # Alanı güncelle
                merged_indices.add(j)  # Bu parçayı artık işlendi say
                merged_any = True
                # Loop'u kırma, belki başka parçalar da eklenecektir.

        # Yeni oluşturulan (veya orijinal) maskeyi kaydet
        new_entry = sorted_masks[i].copy()
        new_entry["segmentation"] = base_mask
        new_entry["area"] = base_area
        final_masks.append(new_entry)

    return final_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", default="raw_capture3.npy")
    parser.add_argument("--bg", default="background.npy")
    parser.add_argument(
        "--checkpoint",
        default="/home/furkand/dev/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")

    # Merge Ayarları
    parser.add_argument(
        "--merge-iou", type=float, default=0.1, help="Birleşme için temas oranı"
    )
    parser.add_argument(
        "--merge-ioa", type=float, default=0.5, help="İç içe geçme oranı (Kapsama)"
    )

    parser.add_argument("--excl-overlap", type=float, default=0.50)
    parser.add_argument("--depth-thresh", type=float, default=10.0)
    parser.add_argument("--min-area-frac", type=float, default=0.001)

    args = parser.parse_args()

    if not os.path.exists(args.npy):
        raise FileNotFoundError(args.npy)
    if not os.path.exists(args.bg):
        raise FileNotFoundError(args.bg)

    # 1. Yükle
    sample = load_sample_from_npy(args.npy)
    bg_sample = load_sample_from_npy(args.bg)
    img = sample["rgb"]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 2. Background Maskesi
    print("Computing exclusion mask...")
    exclusion, fg_vis = compute_exclusion_from_background(
        sample, bg_sample, user_threshold=args.depth_thresh
    )

    if np.count_nonzero(fg_vis) == 0:
        print("\n[UYARI] Foreground simsiyah!")
    else:
        print(f"Foreground detected pixels: {np.count_nonzero(fg_vis)}")

    # 3. SAM2 Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model = build_sam2(args.cfg, args.checkpoint, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
    )

    # 4. Generate
    print("Running SAM2 Generator...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = mask_generator.generate(img)
    print(f"SAM2 generated {len(masks)} raw masks.")

    # 5. Filtrele (Background)
    masks2 = filter_by_exclusion(masks, exclusion, overlap_thresh=args.excl_overlap)
    print(f"After background filter: {len(masks2)} masks.")

    # 6. BİRLEŞTİRME (YENİ ADIM)
    # Burada NMS yerine Smart Merge kullanıyoruz.
    # Ufak gürültüleri temizle
    H, W = masks2[0]["segmentation"].shape
    min_area = int(H * W * args.min_area_frac)
    masks_clean = [m for m in masks2 if m["area"] > min_area]

    print("Merging fragmented parts...")
    final_masks = smart_merge_masks(
        masks_clean,
        iou_thresh=args.merge_iou,  # Eğer %10 bile değiyorlarsa birleşmeye adaydır
        ioa_thresh=args.merge_ioa,  # Eğer biri diğerinin %50 içindeyse kesin birleştir
    )

    print(f"Final merged objects: {len(final_masks)}")

    # 7. Göster
    segmap = render_segmentation_map(final_masks, img.shape[0], img.shape[1])

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Foreground")
    plt.imshow(fg_vis, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title(f"Final Merged ({len(final_masks)} Objects)")
    plt.imshow(segmap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
