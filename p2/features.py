import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations

## Helper functions ############################################################


def inbounds(shape, indices):
    """
    Input:
        shape -- int tuple containing the shape of the array
        indices -- int list containing the indices we are trying
                   to access within the array
    Output:
        True/False, depending on whether the indices are within the bounds of
        the array with the given shape
    """
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################


class KeypointDetector(object):
    # Implement in child classes
    def detectKeypoints(self, image):
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    """
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    """

    def detectKeypoints(self, image):
        image = image.astype(np.float32)
        image /= 255.0
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):
    def computeHarrisValues(self, srcImage):
        """
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        """
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        # compute the gradient
        dx = ndimage.sobel(srcImage, axis=0)
        dy = ndimage.sobel(srcImage, axis=1)

        # compute the harris matrix
        Ixx = dx**2
        Iyy = dy**2
        Ixy = dx * dy

        # gaussian filter
        A = ndimage.gaussian_filter(Ixx, 0.5)
        B = ndimage.gaussian_filter(Ixy, 0.5)
        C = ndimage.gaussian_filter(Iyy, 0.5)

        # compute the harris score
        harrisImage = (A * C) - (B**2) - 0.1 * ((A + C) ** 2)
        orientationImage = np.degrees(np.arctan2(dx, dy))

        # TODO-BLOCK-END

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        """
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        """
        destImage = np.zeros_like(harrisImage, bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        harrisImage = np.pad(harrisImage, 3, "constant", constant_values=0)
        for i in range(3, harrisImage.shape[0] - 3):
            for j in range(3, harrisImage.shape[1] - 3):
                if harrisImage[i, j] == np.max(
                    harrisImage[i - 3 : i + 4, j - 3 : j + 4]
                ):
                    destImage[i - 3, j - 3] = True

        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        """
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        """
        image = image.astype(np.float32)
        image /= 255.0
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.
        # You will need to implement this function.
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.pt = (x, y)
                f.size = 10
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        detector = cv2.ORB_create()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        """
        image = image.astype(np.float32)
        image /= 255.0
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            for j in range(-2, 3):
                for k in range(-2, 3):
                    if inbounds(grayImage.shape, (x + k, y + j)):
                        desc[i, (j + 2) * 5 + (k + 2)] = grayImage[y + j, x + k]

            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        """
        image = image.astype(np.float32)
        image /= 255.0
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))

            # TODO 5: Compute the transform as described by the feature
            # location/orientation and store in 'transMx.' You will need
            # to compute the transform from each pixel in the 40x40 rotated
            # window surrounding the feature to the appropriate pixels in
            # the 8x8 feature descriptor image. 'transformations.py' has
            # helper functions that might be useful
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            # Get initial transformation matrix
            initial_trans = transformations.get_trans_mx(
                np.array([-f.pt[0], -f.pt[1], 0])
            )

            # Get rotation matrix
            rot = transformations.get_rot_mx(0, 0, -np.radians(f.angle))

            # Get scaling matrix
            scale = transformations.get_scale_mx(0.2, 0.2, 1)

            # Get final translation matrix
            trans_again = transformations.get_trans_mx(np.array([4, 4, 0]))

            # Combine transformation matrices
            transMx = np.dot(trans_again, np.dot(scale, np.dot(rot, initial_trans)))[
                :2, (0, 1, 3)
            ]

            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(
                grayImage, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR
            )

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is negligibly small (which we
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            destImage = destImage.flatten()

            std_dest = np.std(destImage)
            if std_dest >= 1e-5:
                dest_mean = np.mean(destImage)
                desc[i, :] = (destImage - dest_mean) / std_dest
            else:
                desc[i, :] = 0
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        raise NotImplementedError

    # Evaluate a match using a ground truth homography. This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6] * x + h[7] * y + h[8]

        return np.array(
            [(h[0] * x + h[1] * y + h[2]) / d, (h[3] * x + h[4] * y + h[5]) / d]
        )


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        """
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching. This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN

        i = 0
        desc1_len = len(desc1)
        desc2_len = len(desc2)
        while i < desc1_len:
            dis_matrix = np.zeros(desc2_len)
            j = 0
            while j < desc2_len:
                # Calculating and assigning the dis_matrix the euclidean dis
                # between desc1 and desc2 the two feature vectors
                euclidean_dis = spatial.distance.euclidean(desc1[i], desc2[j])
                dis_matrix[j] = euclidean_dis
                j = j + 1

            min_axis = np.argmin(dis_matrix)
            matches.append(cv2.DMatch(i, min_axis, dis_matrix[min_axis]))
            i = i + 1

        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        """
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image. If the SSD distance is negligibly small, in this case less
        # than 1e-5, then set the distance to 1. If there are less than two features,
        # set the distance to 0.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN

        i = 0
        desc1_len = len(desc1)
        desc2_len = len(desc2)
        while i < desc1_len:
            dis_matrix = np.zeros(desc2_len)
            j = 0
            while j < desc2_len:
                # Calculating and assigning the dis_matrix the euclidean dis
                # between desc1 and desc2 the two feature vectors
                euclidean_dis = spatial.distance.euclidean(desc1[i], desc2[j])
                dis_matrix[j] = euclidean_dis
                j = j + 1

            min_axis = np.argmin(dis_matrix)
            shortest_dis = dis_matrix[min_axis]

            element = np.argpartition(dis_matrix, 2)[1]
            next_shortest = dis_matrix[element]
            dis_ratio = shortest_dis / next_shortest

            matches.append(cv2.DMatch(i, min_axis, dis_ratio))

            i = i + 1
        # TODO-BLOCK-END

        return matches
