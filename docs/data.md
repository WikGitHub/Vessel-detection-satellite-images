## Background on the Dataset

The aim of this dataset is to facilitate the task of detecting the locations of vessels in satellite images. The automation of this process has wide-ranging applications, including monitoring port activity levels and conducting supply chain analyses.

This dataset comprises images extracted from PlanetScope satellite imagery collected over the San Francisco Bay and San Pedro Bay areas of California. It contains 4,000 RGB images, each with dimensions 80x80 pixels, and is accompanied by annotations categorizing them as either "ship" or "no-ship".

### Key Attributes

- **Label**: Binary classification represented by 1 (ship) or 0 (no-ship).
- **Scene ID**: Universally unique identifier (UUID) denoting the corresponding PlanetScope scene.
- **Longitude_Latitude**: Geospatial coordinates indicating the longitude and latitude of the image.

The pixel values for each 80x80 RGB image are encoded as a list of 19,200 integers, with the first 6,400 entries representing red channel values, the next 6,400 representing green, and the final 6,400 representing blue.

## Classification Categories

The "ship" class comprises 1,000 images, each focusing on a single ship's structure. This category encompasses ships of various sizes, orientations, and atmospheric conditions.

On the other hand, the "no-ship" class consists of 3,000 images, which can be further divided into three subcategories:
- About 1,000 images showcase diverse land cover features such as water bodies, vegetation, bare earth, buildings, and more.
- Another 1,000 images depict partial ship structures, showing only a portion of a ship.
- The remaining 1,000 images have been identified as previously mislabeled by machine learning models due to factors like bright pixels or strong linear features.

## Scenes for Model Evaluation

The dataset also includes full-scene images, providing a visualization tool to assess the performance of classification models trained on this dataset.
