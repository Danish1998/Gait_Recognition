from __future__ import division
from __future__ import print_function

from collections import deque

import numpy as np
import cv2


class Tracker:
    def __init__(self, name=None, buffer=64):
        self.name = name
        self.current_frame = None
        self.previous_frame = None
        self.image_size = None
        self.trackBox = None
        self.roi = None
        self.centroid_pts = deque(maxlen=buffer)
        self.velocity = None

    
    # Not Complete
    def isPerson(self, image, trackBox):
        """Checks if the ROI contains a person
        Args:
            image: input image
            trackbox: a ROI where the target is expected to be
        Returns:
            personDetected: whether or not a person is detected in the ROI
        """

        return True


    def getMatches(self, image1, image2, roi=None, maxFeatures=500, matchPercent=1.0):
        """Calculates the promising matching features between two input images
        Args:
            image1: first input image
            image2: second input image
            maxFeatures: maximum number of features to calculate
            matchPercent: percentage of matches to be considered as good
        Returns:
            matches: the matching features between both input images
            points1: positions of matching features in first image
            points2: positions of corresponding matching features in second image
        """
        # Convert to grayscale if color image
        if not image1.shape[2] == 1:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if not image2.shape[2] == 1:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        if roi is None:
            roi = np.array([0, 0, image1.shape[0], image1.shape[1]])
        
        # Detect ORB features & compute descriptors
        orb = cv2.ORB_create(maxFeatures)
        roi1 = image1[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]]
        roi2 = image2[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]]
        keypoints1, descriptor1 = orb.detectAndCompute(roi1, None)
        keypoints2, descriptor2 = orb.detectAndCompute(roi2, None)

        # Match features
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptor1, descriptor2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * matchPercent)
        matches = matches[:numGoodMatches]

        # Check for tracking failure
        status = True
        if numGoodMatches < 10:
            status = False

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2))
        points2 = np.zeros((len(matches), 2))

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        return status, np.int_(points1), np.int_(points2)


    def setTrackBox(self, image, minBoxAreaFactor=2, windowName=None):
        """Asks the user to set a track box around the target, and checks if it is a person
        Args:
            image: input image containing the target
            windowName: name of the window where the selection is made
        Returns:
            trackBox: a ROI containing the target
        """
        # Set window parameters
        windowSize = int(0.5*(image.shape[0] + image.shape[1]))
        minBoxArea = minBoxAreaFactor*windowSize
        if windowName is None:
            windowName = "Set Tracker"
            
        # Set font properties
        fontScale = windowSize/500.0
        thickness = max(int(windowSize/300.0), 1)
        textOrg = int(windowSize/25.0)
        
        # Open selection window
        cv2.imshow(windowName, image)

        # Set a Track rectangle on the target
        trackBox = None
        image_ = np.copy(image)

        while(True):
            trackBox = cv2.selectROI(windowName, image_, showCrosshair=False, fromCenter=False)
            image_ = np.copy(image)

            # In case no or very small box is selected 
            if trackBox[2]*trackBox[3] < minBoxArea:
                # Display warning message
                cv2.putText(image_, "SET TRACKER ON THE TARGET", (textOrg, textOrg),
                cv2.FONT_HERSHEY_PLAIN, fontScale, (20, 255, 57), thickness)
                continue
            
            # Check if the ROI contains a person
            if not self.isPerson(image, trackBox):
                # Display warning message
                cv2.putText(image_, "CANNOT DETECT ANY PERSON ON THE SELECTED REGION", (textOrg, textOrg),
                 cv2.FONT_HERSHEY_PLAIN, fontScale, (20, 255, 57), thickness)
            else:
                # End loop if a person is selected
                cv2.destroyWindow(windowName)
                break
        
        trackBox = (trackBox[1], trackBox[0], trackBox[3], trackBox[2])
        return trackBox


    def getTrackBox(self, trackBox, velocity, image_size):
        pt1y = max(int(trackBox[0]+velocity[0]), 0)
        pt1x = max(int(trackBox[1]+velocity[1]), 0)
        pt2y = min(int(trackBox[0]+trackBox[2]+velocity[0]), image_size[0])
        pt2x = min(int(trackBox[1]+trackBox[3]+velocity[1]), image_size[1])

        new_trackBox = (pt1y, pt1x, pt2y-pt1y, pt2x-pt1x)
        return new_trackBox


    def initialize(self, image, trackBox=None):
            # Set image as current frame
            self.current_frame = image

            # Get image size
            self.image_size = (image.shape[0], image.shape[1])

            # Set track box around target
            if trackBox is None:
                self.trackBox = self.setTrackBox(self.current_frame)
            else:
                self.trackBox = trackBox

            # Set ROI
            self.roi = self.trackBox
            
            # Get centroid and append it to the queue
            centroid = (int(self.trackBox[0] + 0.5*self.trackBox[2]),
                int(self.trackBox[1] + 0.5*self.trackBox[3]))
            self.centroid_pts.append(centroid)

            # Set initial velocity to be zero
            self.velocity = np.array([0,0])
        

    def update(self, image):
        # Swap the frames
        self.previous_frame = np.copy(self.current_frame)
        self.current_frame = image

        # Get the displacement of target
        ret, points1, points2 = self.getMatches(self.previous_frame, self.current_frame, self.roi)
        if ret:
            displacements = points2 - points1
            self.velocity = np.flip(np.int_(np.mean(displacements, axis=0)), 0) * 4
        else:
            print('Track failure')
            self.velocity = np.array([0,0])

        # Update track box and roi
        self.trackBox = self.getTrackBox(self.trackBox, self.velocity, self.image_size)
        self.roi = self.trackBox

        # Update centroid_pts
        centroid = (int(self.trackBox[0] + 0.5*self.trackBox[2]),
            int(self.trackBox[1] + 0.5*self.trackBox[3]))
        self.centroid_pts.append(centroid)