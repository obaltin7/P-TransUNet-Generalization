# P-TransUNet Generalization Study: ETIS-Larib

This project evaluates the **generalization capability** and robustness of the **P-TransUNet** architecture on the challenging **ETIS-Larib Polyp DB** dataset.

For the original validation study (trained on Kvasir-SEG), please refer to: [P-TransUNetValidation](https://github.com/obaltin7/P-TransUNetValidation)

## 🎯 Project Goal & Challenges

* **Objective:** To test how well the P-TransUNet model performs on a dataset it has never seen before, specifically one known for small and difficult-to-detect polyps.
* **The Challenge:**
    * **Small Dataset:** Only 196 images are available, which is typically insufficient for training Deep Learning models from scratch.
    * **Hard Samples:** ETIS-Larib contains small, flat, and subtle polyps that are harder to segment compared to datasets like Kvasir or CVC-ClinicDB.
* **Our Solution:** To overcome the data scarcity and "data leakage" risks, we implemented a robust pipeline using **Transfer Learning**, **Advanced Data Augmentation**, and **Threshold Tuning**.

## ⚙️ Methodology

To achieve scientifically valid and high-performance results, the following strategies were applied:

1.  **Transfer Learning:** Instead of training from random initialization, we used weights pre-trained on the **Kvasir-SEG** dataset. This provided the model with prior knowledge of polyp features.
2.  **Strict Data Splitting:** To prevent Data Leakage, the dataset was physically split into **Train (166 images)** and **Test (30 images)** folders. The model never saw the Test set during training.
3.  **Advanced Augmentation:** We used the `albumentations` library to apply heavy augmentations (Rotation, Gaussian Noise, RGB Shift, Flip) to simulate a larger dataset and prevent overfitting.
4.  **Threshold Optimization:** During inference, we analyzed different threshold values and determined that **0.3** provides the best balance between Precision and Recall for this specific dataset.

## 📂 Dataset Structure

The **ETIS-Larib** dataset was organized as follows to ensure a deterministic evaluation:

* **Train Set:** 166 Images & Masks (Used for training with augmentation)
* **Test Set:** 30 Images & Masks (Used for final evaluation only)

> **Note:** Due to licensing, the dataset images are not included in this repo. You must download them from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/etis-larib-polyp-database) and arrange them into `data/ETIS-Larib/train` and `data/ETIS-Larib/test`.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/obaltin7/P-TransUNet-Generalization.git](https://github.com/obaltin7/P-TransUNet-Generalization.git)
    cd P-TransUNet-Etis-Generalization
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model (with Transfer Learning):**
    ```bash
    python train.py
    ```

4.  **Evaluate the model:**
    ```bash
    python test.py
    ```
    *This script will calculate metrics using the optimized threshold (0.3) and save the best visual results to the `output/` folder.*

## 📊 Experimental Results

The model was fine-tuned for 50 epochs on an **NVIDIA RTX 2060 (6GB)**. Despite the limited data, the combination of Transfer Learning and Augmentation significantly boosted performance compared to the baseline.

| Metric | Baseline (From Scratch) | **Final Model (Transfer Learning + Aug + Thresh 0.3)** |
| :--- | :--- | :--- |
| **mDice** | ~0.4200 | **0.6121** |
| **mIoU** | ~0.3500 | **0.5276** |
| **Recall** | ~0.3800 | **0.6517** |
| **Precision**| ~0.6500 | **0.6615** |

> **Analysis:** The baseline model struggled to detect small polyps (Recall ~38%). With our optimizations, **Recall increased to 65%**, meaning the model successfully captures the majority of difficult polyp cases.

## ⚖️ License & Citation

This project is an unofficial implementation for research purposes using the P-TransUNet architecture.

If you use this code for your research, please cite the original paper:
> Chong, Yan-Wen & Xie, Ningdi & Liu, Xin & Pan, Shaoming. (2023). P-TransUNet: an improved parallel network for medical image segmentation. BMC Bioinformatics. 24. 10.1186/s12859-023-05409-7.
