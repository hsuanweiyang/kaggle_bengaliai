# Kaggle Bengaliai competition

*   Description:  For this competition, you’re given the image of a handwritten Bengali grapheme and are
challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

    *   Public Leaderboard: 0.9457/0.9955(#1)
    *   Private Leaderboard: 0.8984/0.9762(#1)

## Image Preprocessing
### 1. Raw Image (137*236)
    *   <img src="data_description/sample_1-0.png">

### 2. Thresholding
<img src="data_description/sample_1-1.png">

### 3. Edge Detection (Canny)
<img src="data_description/sample_1-2.png">

### 4. Region of Interest
<img src="data_description/sample_1-3.png">

### 5. Cut & Resize (64*64)
<img src="data_description/sample_1-4.png">
<img src="data_description/sample_1-5.png">

## Model

### Light version of ResNet18
<img src="model/resnet-18-light.png">
