import os
import re
import hashlib
from collections import defaultdict

import fitz
from rtree import index


def extract_pdf_ragntex(pdf_path: str, article_dir: str):
  images_dir = os.path.join(article_dir, "ragntex")
  os.makedirs(images_dir,  exist_ok=True)

  doc = fitz.open(pdf_path)

  text = ""
  for page_num, page in enumerate(doc):
    # Parse the text
    text = " ".join([text, page.get_text().strip()])

    # Extract images
    extract_images(doc, page, page_num, images_dir)

    # Extract vector graphics
    extract_vector(doc, page, page_num, images_dir)

  # Save the text
  text_path = os.path.join(article_dir, "ragntex.txt")
  with open(text_path, "w", encoding="utf-8") as f:
    f.write(text)
  f.close()

  return True


def extract_images(doc, page, page_num, images_dir):
  images = page.get_images(full=True)

  for img_index, img in enumerate(images):
    # Extract the image
    xref = img[0]
    base_image  = doc.extract_image(xref)
    image_bytes = base_image["image"]
    image_hash  = hashlib.md5(image_bytes).hexdigest()
    image_name  = f"page{page_num}_img{img_index}_hash{image_hash[:8]}.png"

    # Save the image
    image_path = os.path.join(images_dir, image_name)
    with open(image_path, "wb") as f:
      f.write(image_bytes)
    f.close()

  return True


def are_bounding_boxes_close(bbox1, bbox2, threshold=50):
    # Extracting the four edges of each bounding box
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # Check if any of the borders are within the threshold distance
    return (
        abs(left1 - right2) < threshold
        or abs(right1 - left2) < threshold
        or abs(top1 - bottom2) < threshold
        or abs(bottom1 - top2) < threshold
    )


def merge_bounding_boxes(bboxes):
    if not bboxes:
        return None
    # Start with the first bounding box
    combined_bbox = bboxes[0]
    for bbox in bboxes[1:]:
        combined_bbox = combined_bbox | bbox  # Combine the bounding boxes (union)
    return combined_bbox


def group_bounding_boxes(bboxes, max_drawings=2000, threshold=50):
    # R-tree index setup
    idx = index.Index()
    for i, rect in enumerate(bboxes):
        expanded = rect + (-threshold, -threshold, threshold, threshold)
        idx.insert(i, expanded)

    # Graph connectivity
    adj_list = defaultdict(list)
    for i, rect in enumerate(bboxes):
        expanded = rect + (-threshold, -threshold, threshold, threshold)
        for j in idx.intersection(expanded):
            if i != j:
                adj_list[i].append(j)

    # Perform DFS to find connected components (groups of connected bounding boxes)
    visited = [False] * len(bboxes)
    components = []

    def dfs(node, component):
        visited[node] = True
        component.append(bboxes[node])
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)

    # Find all connected components using DFS
    for i in range(len(bboxes)):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    # Return grouped and merged bboxes
    return [merge_bounding_boxes(group) for group in components]


def process_large_drawing(drawings, max_drawings=1000, threshold=50):
    bboxes = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]

    if len(bboxes) < max_drawings:
        return group_bounding_boxes(bboxes, threshold=threshold)

    # Split the data into smaller chunks
    num_chunks = (len(bboxes) // max_drawings) + 1
    all_results = []

    for chunk_index in range(num_chunks):
        chunk = bboxes[chunk_index * max_drawings : (chunk_index + 1) * max_drawings]
        results = group_bounding_boxes(chunk, threshold=threshold)
        all_results.extend(results)

    # Return the combined results
    return group_bounding_boxes(all_results, threshold=threshold)


def find_surrounding_text(page, group, threshold=50):
    text_blocks = page.get_text("dict")["blocks"]
    expanded = group + (-threshold, -threshold, threshold, threshold)
    surrounding = []

    for block in text_blocks:
        if block["type"] != 0:
            continue

        block_rect = fitz.Rect(block["bbox"])
        if expanded.intersects(block_rect):
            surrounding.append(block_rect)

    return surrounding


def extract_vector(doc, page, page_num, images_dir):
  MAX_DRAWINGS = 1000
  MIN_SIZE = 0.05
  MAX_SIZE = 0.30
  THRESHOLD = 5
  ZOOM = 4

  page_size = page.rect.width * page.rect.height
  min_size  = page_size * MIN_SIZE
  max_size  = page_size * MAX_SIZE

  drawings = page.get_drawings()

  # Group drawings into figures
  grouped = process_large_drawing(
    drawings, max_drawings=MAX_DRAWINGS, threshold=THRESHOLD
  )

  for group_num, group in enumerate(grouped):
    # Try to include any text labels around
    surrounding = find_surrounding_text(page, group, threshold=THRESHOLD)
    if surrounding:
      figure_bbox = merge_bounding_boxes([group] + surrounding)
    else:
      figure_bbox = group

    # Filter by minimal plot size
    width = figure_bbox[2] - figure_bbox[0]
    height = figure_bbox[3] - figure_bbox[1]
    area = width * height
    if area > min_size and area < max_size:
      scale_mat = fitz.Matrix(ZOOM, ZOOM)
      figure_pix = page.get_pixmap(matrix=scale_mat, clip=figure_bbox)
      figure_bytes = figure_pix.tobytes("png")
      figure_hash = hashlib.md5(figure_bytes).hexdigest()
      figure_name = f"page{page_num}_fig{group_num}_hash{figure_hash[:8]}.png"

      figure_path = os.path.join(images_dir, figure_name)
      with open(figure_path, "wb") as f:
        f.write(figure_bytes)
      f.close()

  return True
