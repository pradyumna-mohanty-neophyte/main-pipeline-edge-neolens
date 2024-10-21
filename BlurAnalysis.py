import cv2
import numpy as np
# import matplotlib.pyplot as plt

class BlurAnalysis:
    def __init__(self):
        """
        Initializes the BlurAnalysis class.
        """
        pass

    def tenengrad_sharpness(self, image):
        """
        Calculate the Tenengrad sharpness score using the Sobel operator.

        Args:
            image (np.ndarray): Input image in grayscale.

        Returns:
            float: Sharpness score based on the Sobel gradient.
        """
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sharpness_score = np.mean(gradient_magnitude)
        return sharpness_score

    def estimate_optimal_threshold(self, scores):
        """
        Estimate the optimal threshold value using the mean and standard deviation of the scores.

        Args:
            scores (list): List of sharpness scores.

        Returns:
            float: Estimated optimal threshold for classifying images.
        """
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        optimal_threshold = mean_score - 0.5 * std_score

        # Display the distribution of scores
        # plt.figure(figsize=(10, 6))
        # plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
        # plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Estimated Threshold = {optimal_threshold:.2f}')
        # plt.xlabel('Sharpness Scores')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Sharpness Scores')
        # plt.legend()
        # plt.show()

        return optimal_threshold

    def estimate(self, cropped_images):
        """
        Estimate the optimal sharpness threshold (bandwidth) based on a set of cropped images.
        This is like a "training" or calibration mode.

        Args:
            cropped_images (list of np.ndarray): List of cropped images in BGR format.

        Returns:
            float: The estimated optimal threshold for classifying images as blurry or non-blurry.
        """
        sharpness_scores = []

        # Loop through all cropped images
        for cropped_image in cropped_images:
            # Convert the cropped image to grayscale and calculate sharpness score
            cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            score = self.tenengrad_sharpness(cropped_image_gray)
            sharpness_scores.append(score)

        # Calculate the optimal threshold based on the sharpness scores
        if sharpness_scores:
            optimal_threshold = self.estimate_optimal_threshold(sharpness_scores)
            print(f"Estimated Optimal Threshold (Calibration): {optimal_threshold:.2f}")
            return optimal_threshold
        else:
            print("No valid images found to estimate the threshold.")
            return None

    def predict(self, cropped_image, threshold):
        """
        Classify a cropped image as blurry or non-blurry in live inference mode based on the provided threshold.

        Args:
            cropped_image (np.ndarray): The cropped image in BGR format.
            threshold (float): The threshold value to classify the image as blurry or non-blurry.

        Returns:
            str: Classification result ('Blurry' or 'Non-Blurry').
        """
        # Convert the cropped image to grayscale and calculate sharpness score
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        sharpness_score = self.tenengrad_sharpness(cropped_image_gray)

        # Classify based on the threshold
        classification = 'Blurry' if sharpness_score < threshold else 'Non-Blurry'
        label_color = (0, 0, 255) if classification == "Blurry" else (0, 255, 0)
        label_text = f"{classification}: {sharpness_score:.2f}"
        # cv2.putText(cropped_image, label_text, (10, 30), cv2.F    ONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)

        # Display the image with classification
        # plt.figure(figsize=(6, 6))
        # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"{classification} | Score: {sharpness_score:.2f}", color='red' if classification == 'Blurry' else 'green')
        # plt.axis("off")
        # plt.show()

        return classification
