import numpy as np


class HistoryKeeper(object):
    """
    Keeps track of the windows for the left and right lane lines in each frame.
    """
    def __init__(self, window_width, window_height, margin, smooth_factor=15):
        self.recent_centers = []  # History of the centers previously found
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.smooth_factor = smooth_factor  # Size of the slice of recent centers to be considered when smoothing the window centroids

    def find_window_centroids(self, warped_image):
        """
        We take a warped image and slice it in horizontal stripes where we'll determine the regions with most pixels intensity
        for both the left and right section of each stripe. This process will give us the X coordinate location where the most pixel
        intensity happens in the left region of a given stripe, and analogously for the right region.

        Finally, we'll calculate the average left and right location (centroid) over the last `smooth_factor` centroids.

        This idea was hugely inspired in the solution presented by this Udacity's instructor in his live Q&A session:
        - https://www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be  [Amazing!]

        Here are some useful links to understand the tricky convolution concept (which is kinda hard btw):
        - https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
        - https://www.khanacademy.org/math/differential-equations/laplace-transform/convolution-integral/v/introduction-to-the-convolution
        - https://en.wikipedia.org/wiki/Convolution
        """
        window_centroids = []
        window = np.ones(self.window_width)
        image_width, image_height = warped_image.shape[1], warped_image.shape[0]

        window_width_center = self.window_width / 2  # This is the center of the image in the x axis
        stripe_height_boundary = int(3 * image_height / 4)
        stripe_width_boundary = int(image_width / 2)
        first_stripe_left_half = warped_image[stripe_height_boundary:, :stripe_width_boundary]
        left_sum = np.sum(first_stripe_left_half, axis=0)  # Calculates the number of pixels per column in the left half of the stripe
        left_center = np.argmax(np.convolve(window, left_sum)) - window_width_center  # Get the location and shift if to the center of the window

        first_stripe_right_half = warped_image[stripe_height_boundary:, stripe_width_boundary:]
        right_sum = np.sum(first_stripe_right_half, axis=0)  # Calculates the number of pixels per column in the right half of the stripe
        right_center = np.argmax(np.convolve(window, right_sum)) - window_width_center + stripe_width_boundary  # Get the location and shift if to the center of the window

        new_centroid = (left_center, right_center)
        window_centroids.append(new_centroid)

        # Repeat for the remaining stripes
        number_of_windows = int(image_height / self.window_height)
        for level in range(1, number_of_windows):
            stripe_height_boundary = int(image_height - (level + 1) * self.window_height)
            stripe_width_boundary = int(image_height - level * self.window_height)

            image_layer = np.sum(warped_image[stripe_height_boundary:stripe_width_boundary, :], axis=0)
            conv_signal = np.convolve(window, image_layer)

            offset = self.window_width / 2
            left_lower_bound = int(max(left_center + offset - self.margin, 0))
            left_max_bound = int(min(left_center + offset + self.margin, image_width))
            left_center = np.argmax(conv_signal[left_lower_bound:left_max_bound]) + left_lower_bound - offset

            right_lower_bound = int(max(right_center + offset - self.margin, 0))
            right_upper_bound = int(min(right_center + offset + self.margin, image_width))
            right_center = np.argmax(conv_signal[right_lower_bound:right_upper_bound]) + right_lower_bound - offset

            new_centroid = (left_center, right_center)
            window_centroids.append(new_centroid)

        self.recent_centers.append(window_centroids)

        # We take into account the last N centers to prevent wobbling and irregularities.
        most_recent_centers = self.recent_centers[-self.smooth_factor:]
        return np.mean(most_recent_centers, axis=0)
