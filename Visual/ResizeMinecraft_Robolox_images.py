import cv2
import os

# ─────────────────────────────────────────────────
# CONFIG — point these to your folders
# ─────────────────────────────────────────────────
IMG_SIZE = 260

FOLDERS_TO_RESIZE = [
    r"D:\MineCraft and robolox safe train",
    r"D:\MineCraft and robolox safe vall",

]

# ─────────────────────────────────────────────────
# RESIZE ALL IMAGES IN A FOLDER
# Overwrites original images with resized versions
# ─────────────────────────────────────────────────
def resize_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"  ❌ Folder not found: {folder_path}")
        return 0

    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"  ⚠️  No images found in: {folder_path}")
        return 0

    print(f"\n  📂 {folder_path}")
    print(f"  Found {len(images)} images — resizing to {IMG_SIZE}x{IMG_SIZE}...")

    resized_count = 0
    skipped_count = 0

    for i, filename in enumerate(images):
        img_path = os.path.join(folder_path, filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Could not read: {filename}")
            skipped_count += 1
            continue

        # Check if already correct size
        h, w = img.shape[:2]
        if h == IMG_SIZE and w == IMG_SIZE:
            skipped_count += 1
            continue

        # Resize and overwrite
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(img_path, resized)
        resized_count += 1

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(images)}")

    print(f"  ✅ Resized : {resized_count}")
    print(f"  ⏭️  Skipped : {skipped_count} (already correct size or unreadable)")
    return resized_count


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(f"   RESIZING IMAGES TO {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 55)

    total_resized = 0
    for folder in FOLDERS_TO_RESIZE:
        total_resized += resize_folder(folder)

    print("\n========== SUMMARY ==========")
    print(f"  Total resized: {total_resized} images")
    print(f"  All images are now {IMG_SIZE}x{IMG_SIZE}")
    print("✅ Done!")