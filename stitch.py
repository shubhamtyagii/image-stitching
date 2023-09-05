import numpy as np
import cv2
import random
import math

NO_OF_POINTS = 100

class Stitcher:
    def __init__(self, images, offset = [500,500], final_image_dims = [2000,6000], target_idx = 0):
        '''
            images: List of images where 
            assumption is images[i] stitches with images[i+1]
        '''
        self.images = images
        self.final_image = self.create_warp_plane(final_image_dims)
        self.offset = offset
        self.tmp = 0
        self.target_img_idx = target_idx
        self.homography_matrices = [None for i in range(len(self.images))]
        self.homography_matrices[self.target_img_idx] = np.eye(3)

    
    def extract_SIFT_features(self,image):
        '''
            Extract features from the image
        '''

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints,descriptors

    def prep_A(self, kp1,kp2,matches):
        '''
            Prepare matrix to calculate the homography.
        '''

        A = np.array([])
        for match in matches:
            xs,ys = kp1[match.queryIdx].pt
            xd,yd = kp2[match.trainIdx].pt
            row1 = np.array([xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd])
            row2 = np.array([0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd])
            A = np.append(A, row1)
            A = np.append(A, row2)
        A = np.reshape(A,(-1, 9))
        assert A.shape[0] == 2*len(matches) and A.shape[1] == 9
        return A

    def create_warp_plane(self,d):
        '''
            Create an image big enough to contain the panaroma.
        '''

        # To do 
        return np.zeros(shape=(d[0],d[1],3),dtype=np.uint8)
    
    def match_points(self, d1, d2):
        bf = cv2.BFMatcher()
        matches = bf.match(d1, d2)
        return sorted(matches, key = lambda x:x.distance)
    
    def run_RANSAC(self, matches, kp1, kp2, threshold = 0.5, no_of_trials = 50, no_of_samples = 15):
        '''
            Run RANSAC algorithm to get the Homography matrix which maps the image[i+1] best to image[i]
        '''

        
        max_aligned_points = 0
        final_H = None
        
        for _ in range(no_of_trials):
            t_matches = []
            samples = random.sample(range(NO_OF_POINTS), no_of_samples)
            for idx in samples:
                t_matches.append(matches[idx])
                A = self.prep_A(kp1, kp2, t_matches)
                A_ = np.matmul(A.T, A)
                _, _, V = np.linalg.svd(A_)
                H = np.reshape(V[-1], (3, 3))
                # check how well this matrix is doing
                curr_alignment_count = 0
                for match in matches:
                    xs,ys = kp1[match.queryIdx].pt
                    dst = np.append(kp2[match.trainIdx].pt,1).T
                    t = np.dot(H, np.array([xs, ys, 1]))
                    t /= t[2]
                    if np.linalg.norm(t - dst) < threshold:
                        curr_alignment_count += 1

                if curr_alignment_count > max_aligned_points or final_H is None:
                    max_aligned_points = curr_alignment_count
                    final_H = H
        print('max found aligned points:', max_aligned_points)
        return final_H

    def calculate_homography(self, img_idx, target_img_idx):
        target_image = self.images[target_img_idx]
        image_2 = self.images[img_idx]
        target_image = cv2.cvtColor(target_image,cv2.COLOR_RGB2GRAY)
        image_2 = cv2.cvtColor(image_2,cv2.COLOR_RGB2GRAY)
        kp1, d1 = self.extract_SIFT_features(image_2)
        kp2, d2 = self.extract_SIFT_features(target_image)
        matches = self.match_points(d1, d2)[:NO_OF_POINTS] # taking top matches

        # custom homography calc 
        H = self.run_RANSAC(matches, kp1, kp2)
        return H
    
    
    def stitch(self):
        for idx in range(self.target_img_idx, len(self.images)):

            if idx == self.target_img_idx:
                self.warp_image_custom(idx)
            
            else:
                H = self.calculate_homography(idx, idx-1)
                self.homography_matrices[idx] = np.matmul(self.homography_matrices[idx-1], H)
                self.warp_image_custom(idx)

        for idx in range(self.target_img_idx-1,-1,-1):
            H = self.calculate_homography(idx, idx+1)
            self.homography_matrices[idx] = np.matmul(self.homography_matrices[idx+1], H)
            self.warp_image_custom(idx)
        
        return self.final_image
        
    def warp_image_custom(self, idx, forward_warp = False):
        '''
            Forward warp the image using the homography matrix
        '''
        H = self.homography_matrices[idx]
        img = self.images[idx]
        h, w, _ =  img.shape
        if forward_warp:
            coords = np.indices((w, h)).reshape(2, -1)
            coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(int)    
            transformedPoints = np.dot(H, coords)
            yo, xo = coords[1, :], coords[0, :]
            # projective transform. Output's 3rd index should be one to convert to cartesian coords.
            yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(int)
            xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(int)
            self.final_image[yt + self.offset[0], xt + self.offset[1]] = img[yo, xo]
            cv2.imwrite('warped_'+str(self.tmp)+'.jpg', self.final_image)
            self.tmp +=1
        else:          
            corners = np.array([[0,0,1],[0,h-1,1],[w-1,h-1,1],[w-1,0,1]])
            transformedPoints = np.dot(H, corners.T)

            H_inv = np.linalg.inv(H)
            y_corners = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(int)
            x_corners = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(int)
            top_left = [y_corners[0],x_corners[0]]
            bottom_left = [y_corners[1],x_corners[1]]
            bottom_right = [y_corners[2],x_corners[2]]
            top_right = [y_corners[3],x_corners[3]]
            print('top_left:',top_left,' top_right:', top_right ,' bottom_right:', bottom_right,' bottom_left:',bottom_left)
            width_start = min(top_left[1], bottom_left[1])
            width_end =  max(top_right[1], bottom_right[1])

            height_start = min(top_left[0], top_right[0])
            height_end = max(bottom_right[0], bottom_left[0])

            coords = np.indices((width_end - width_start, height_end - height_start)).reshape(2, -1)
            coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(int)
            coords[0, :] += width_start
            coords[1, :] += height_start
            h_o, w_o = coords[1, :], coords[0, :]

            transformed_points = np.dot(H_inv, coords)
            h_i = np.divide(np.array(transformed_points[1, :]),np.array(transformed_points[2, :])).astype(int)
            w_i = np.divide(np.array(transformed_points[0, :]),np.array(transformed_points[2, :])).astype(int)
            indices = np.where((h_i >= 0) & (h_i < h) & (w_i >= 0) & (w_i < w))
            w_o = w_o[indices]
            h_o = h_o[indices]
            w_i = w_i[indices]
            h_i = h_i[indices]

            self.final_image[h_o + self.offset[0], w_o + self.offset[1]] = img[h_i, w_i]

if __name__=='__main__':
    i = 1
    l = []
    for i in range(1,4):
        l.append(cv2.imread('dataset/2/weir_'+str(i)+'.jpg'))
    s = Stitcher(l, target_idx=1, offset=[500,1000])
    stitched_image = s.stitch()
    cv2.imwrite('stitched.jpg', stitched_image )