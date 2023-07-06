import cv2
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import imageio
import imageio.v2 as imageio


class LaneFilter:
  def __init__(self, p):
    self.sat_thresh = p['sat_thresh']
    self.light_thresh = p['light_thresh'] 
    self.light_thresh_agr = p['light_thresh_agr']
    self.grad_min, self.grad_max = p['grad_thresh']
    self.mag_thresh, self.x_thresh = p['mag_thresh'], p['x_thresh']
    self.hls, self.l, self.s, self.z  = None, None, None, None
    self.color_cond1, self.color_cond2 = None, None
    self.sobel_cond1, self.sobel_cond2, self.sobel_cond3 = None, None, None 

  def sobel_breakdown(self, img):
    self.apply(img)
    b1, b2, b3 = self.z.copy(), self.z.copy(), self.z.copy()
    b1[(self.sobel_cond1)] = 255
    b2[(self.sobel_cond2)] = 255
    b3[(self.sobel_cond3)] = 255
    return np.dstack((b1, b2,b3))

  def color_breakdown(self, img):
    self.apply(img)
    b1, b2 = self.z.copy(), self.z.copy()
    b1[(self.color_cond1)] = 255
    b2[(self.color_cond2)] = 255
    return np.dstack((b1, b2, self.z))

  def apply(self, rgb_image):    
    self.hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    self.l = self.hls[:, :, 1]
    self.s = self.hls[:, :, 2]
    self.z = np.zeros_like(self.s)
    color_img = self.apply_color_mask()
    sobel_img = self.apply_sobel_mask()
    filtered_img = cv2.bitwise_or(sobel_img, color_img)
    return filtered_img

  def apply_color_mask(self):   
    self.color_cond1 = (self.s > self.sat_thresh) & (self.l > self.light_thresh)
    self.color_cond2 = self.l > self.light_thresh_agr
    b = self.z.copy()
    b[(self.color_cond1 | self.color_cond2)] = 1
    return b

  def apply_sobel_mask(self):       
    lx = cv2.Sobel(self.l, cv2.CV_64F, 1, 0, ksize = 5)
    ly = cv2.Sobel(self.l, cv2.CV_64F, 0, 1, ksize = 5)
    gradl = np.arctan2(np.absolute(ly), np.absolute(lx))
    l_mag = np.sqrt(lx**2 + ly**2)
    slm, slx, sly = scale_abs(l_mag), scale_abs(lx), scale_abs(ly)
    b = self.z.copy()
    self.sobel_cond1 = slm > self.mag_thresh
    self.sobel_cond2 = slx > self.x_thresh
    self.sobel_cond3 = (gradl > self.grad_min) & (gradl < self.grad_max)
    b[(self.sobel_cond1 & self.sobel_cond2 & self.sobel_cond3)] = 1  
    return b

def scale_abs(x, m = 255):
  x = np.absolute(x)
  x = np.uint8(m * x / np.max(x))
  return x 

def roi(gray, mn = 125, mx = 1200):
  m = np.copy(gray) + 1
  m[:, :mn] = 0 
  m[:, mx:] = 0 
  return m


class Curves:
  def __init__(self, number_of_windows, margin, minimum_pixels, ym_per_pix, xm_per_pix):
    
    self.min_pix = minimum_pixels
    self.margin = margin
    self.n = number_of_windows
    self.ky, self.kx = ym_per_pix, xm_per_pix

    self.binary, self.h, self.w, self.window_height = None, None, None, None
    self.all_pixels_x, self.all_pixels_y = None, None
    self.left_pixels_indices, self.right_pixels_indices = [], []
    self.left_pixels_x, self.left_pixels_y = None, None
    self.right_pixels_x, self.right_pixels_y = None, None
    self.out_img = None 
    self.left_fit_curve_pix, self.right_fit_curve_pix = None, None
    self.left_fit_curve_f, self.right_fit_curve_f = None, None
    self.left_radius, self.right_radius = None, None
    self.vehicle_position, self.vehicle_position_words = None, None
    self.result = {}
    
  def store_details(self, binary):
    self.out_img = np.dstack((binary, binary, binary)) * 255
    self.binary = binary
    self.h, self.w = binary.shape[0], binary.shape[1]
    self.mid = self.h / 2
    self.window_height = np.int(self.h / self.n)  
    self.all_pixels_x = np.array(binary.nonzero()[1])
    self.all_pixels_y = np.array(binary.nonzero()[0])
    
  def start(self, binary):
    hist = np.sum(binary[np.int(self.h / 2):, :], axis = 0)
    mid = np.int(hist.shape[0] / 2)
    current_leftx = np.argmax(hist[:mid])
    current_rightx = np.argmax(hist[mid:]) + mid
    return current_leftx, current_rightx

  def next_y(self, w):
    y_lo = self.h - (w + 1) * self.window_height
    y_hi = self.h - w * self.window_height 
    return y_lo, y_hi

  def next_x(self, current):
    x_left = current - self.margin
    x_right = current + self.margin
    return x_left, x_right
  
  def next_midx(self, current, pixel_indices):
    if len(pixel_indices) > self.min_pix:
      current = np.int(np.mean(self.all_pixels_x[pixel_indices]))
    return current

  def draw_boundaries(self, p1, p2, color, thickness = 5):
    cv2.rectangle(self.out_img, p1, p2, color, thickness)
    #cv2.line(self.out_img, self.h, self.w,)

  def indices_within_boundary(self, y_lo, y_hi, x_left, x_right):
    cond1 = (self.all_pixels_y >= y_lo)
    cond2 = (self.all_pixels_y < y_hi)
    cond3 = (self.all_pixels_x >= x_left)
    cond4 = (self.all_pixels_x < x_right)
    return (cond1 & cond2 & cond3 & cond4 ).nonzero()[0]

  def pixel_locations(self, indices):
    return self.all_pixels_x[indices], self.all_pixels_y[indices]
  
  def plot(self, t: object = 4) -> object:
  
    self.out_img[self.left_pixels_y, self.left_pixels_x] = [255, 0, 255]
    self.out_img[self.right_pixels_y, self.right_pixels_x] = [0, 255, 255]

    self.left_fit_curve_pix = np.polyfit(self.left_pixels_y, self.left_pixels_x, 2)
    self.right_fit_curve_pix = np.polyfit(self.right_pixels_y, self.right_pixels_x, 2)

    kl, kr = self.left_fit_curve_pix, self.right_fit_curve_pix
    ys = np.linspace(0, self.h - 1, self.h)
    
    left_xs = kl[0] * (ys**2) + kl[1] * ys + kl[2]
    right_xs = kr[0] * (ys**2) + kr[1] * ys + kr[2]
    
    xls, xrs, ys = left_xs.astype(np.uint32), right_xs.astype(np.uint32), ys.astype(np.uint32)
    
    for xl, xr, y in zip(xls, xrs, ys):
      cv2.line(self.out_img, (xl - t, y), (xl + t, y), (255, 255, 0), int(t / 2))
      cv2.line(self.out_img, (xr - t, y), (xr + t, y), (0, 0, 255), int(t / 2))


  
  def get_real_curvature(self, xs, ys):
    return np.polyfit(ys * self.ky, xs * self.kx, 2)
  
  def radius_of_curvature(self, y, f):
    return ((1 + (2 * f[0] * y + f[1])**2)**(1.5)) / np.absolute(2 * f[0])

  def update_vehicle_position(self):
    y = self.h
    mid = self.w / 2
    kl, kr = self.left_fit_curve_pix, self.right_fit_curve_pix
    xl = kl[0] * (y**2) + kl[1]* y + kl[2]
    xr = kr[0] * (y**2) + kr[1]* y + kr[2]
    pix_pos = xl + (xr - xl) / 2
    self.vehicle_position = (pix_pos - mid) * self.kx 

    if self.vehicle_position < 0:
      self.vehicle_position_words = str(np.absolute(np.round(self.vehicle_position, 2))) + " m left of center"
    elif self.vehicle_position > 0:
      self.vehicle_position_words = str(np.absolute(np.round(self.vehicle_position, 2))) + " m right of center"
    else:
      self.vehicle_position_words = "at the center"

  def fit(self, binary):
    
    self.store_details(binary)
    mid_leftx, mid_rightx = self.start(binary)

    left_pixels_indices, right_pixels_indices = [], []
    x, y = [None, None, None, None], [None, None]
    
    for w in range(self.n):
      
      y[0], y[1] = self.next_y(w)
      x[0], x[1] = self.next_x(mid_leftx) 
      x[2], x[3] = self.next_x(mid_rightx)
        
      self.draw_boundaries((x[0], y[0]), (x[1], y[1]), (255, 0, 0))
      self.draw_boundaries((x[2], y[0]), (x[3], y[1]), (0, 255, 0))

      
      curr_left_pixels_indices = self.indices_within_boundary(y[0], y[1], x[0], x[1])
      curr_right_pixels_indices = self.indices_within_boundary(y[0], y[1], x[2], x[3])
      
      left_pixels_indices.append(curr_left_pixels_indices)
      right_pixels_indices.append(curr_right_pixels_indices)
      
      mid_leftx = self.next_midx(mid_leftx, curr_left_pixels_indices)
      mid_rightx = self.next_midx(mid_rightx, curr_right_pixels_indices)
    
    self.left_pixels_indices = np.concatenate(left_pixels_indices)
    self.right_pixels_indices = np.concatenate(right_pixels_indices)
    
    self.left_pixels_x, self.left_pixels_y = self.pixel_locations(self.left_pixels_indices)
    self.right_pixels_x, self.right_pixels_y = self.pixel_locations(self.right_pixels_indices)

    self.left_fit_curve_f = self.get_real_curvature(self.left_pixels_x, self.left_pixels_y)
    self.right_fit_curve_f = self.get_real_curvature(self.right_pixels_x, self.right_pixels_y)
    
    self.left_radius = self.radius_of_curvature(self.h * self.ky, self.left_fit_curve_f)
    self.right_radius = self.radius_of_curvature(self.h *  self.ky, self.right_fit_curve_f)

    self.plot()
    self.update_vehicle_position()

    self.result = {
      'image': self.out_img,
      'left_radius': self.left_radius,
      'right_radius': self.right_radius,
      'real_left_best_fit_curve': self.left_fit_curve_f,
      'real_right_best_fit_curve': self.right_fit_curve_f, 
      'pixel_left_best_fit_curve': self.left_fit_curve_pix,
      'pixel_right_best_fit_curve': self.right_fit_curve_pix, 
      'vehicle_position': self.vehicle_position, 
      'vehicle_position_words': self.vehicle_position_words
    }

    return self.result


class BirdsEye:
    
  def __init__(self, source_points, dest_points, cam_matrix, distortion_coef):
    self.spoints = source_points
    self.dpoints = dest_points
    self.src_points = np.array(source_points, np.float32)
    self.dest_points = np.array(dest_points, np.float32)
    self.cam_matrix = cam_matrix
    self.dist_coef = distortion_coef
    
    self.warp_matrix = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
    self.inv_warp_matrix = cv2.getPerspectiveTransform(self.dest_points, self.src_points)

  def undistort(self, raw_image, show_dotted = False):
     
    image = cv2.undistort(raw_image, self.cam_matrix, self.dist_coef, None, self.cam_matrix)
    
    if show_dotted: 
      show_dotted_image(image, self.spoints)
        
    return image 

  def sky_view(self, ground_image, show_dotted = False):
    
    temp_image = self.undistort(ground_image, show_dotted = False)
    shape = (temp_image.shape[1], temp_image.shape[0])
    warp_image = cv2.warpPerspective(temp_image, self.warp_matrix, shape, flags = cv2.INTER_LINEAR)
    
    if show_dotted: 
      show_dotted_image(warp_image, self.dpoints)
    
    return warp_image

  def project(self, ground_image, sky_lane, left_fit, right_fit, color = (0, 255, 0)):

    z = np.zeros_like(sky_lane)
    sky_lane = np.dstack((z, z, z))

    kl, kr = left_fit, right_fit
    h = sky_lane.shape[0]
    ys = np.linspace(0, h - 1, h)
    lxs = kl[0] * (ys**2) + kl[1]* ys +  kl[2]
    rxs = kr[0] * (ys**2) + kr[1]* ys +  kr[2]
    
    pts_left = np.array([np.transpose(np.vstack([lxs, ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rxs, ys])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(sky_lane, np.int_(pts), color)
    
    shape = (sky_lane.shape[1], sky_lane.shape[0])
    ground_lane = cv2.warpPerspective(sky_lane, self.inv_warp_matrix, shape)

    result = cv2.addWeighted(ground_image, 1, ground_lane, 0.3, 0)
    return result

def draw_line_n_mark_circle(img, points):
    cv2.line(img, points[0], points[1], (255, 0, 255), 2)
    cv2.line(img, points[2], points[3], (255, 0, 255), 2)
    #cv2.line(img, (int(points[0] / 2), int(points[1] / 2)), (0, 0, 255), 2)
    #cv2.line(img, (int(points[2] / 2), int(points[3] / 2)), (0, 0, 255), 2)

    for point in points:
        cv2.circle(img, point, 5, (255, 0, 0), -1)


def merge_n_resize_4images(img1, img2, img3, img4):
    temp = cv2.hconcat([img1, img2])
    temp = cv2.hconcat([temp, img3])
    temp = cv2.hconcat([temp, img4])
    resized = cv2.pyrDown(temp)
    resized = cv2.pyrDown(resized)
    return resized


##### image load ###########################################
file_path = 'C:\\Users\\DGSW\\OneDrive\\바탕 화면\\Bird_eye_line_detect\\videos\\project_video.mp4'
capture = cv2.VideoCapture(file_path)
fps = capture.get(cv2.CAP_PROP_FPS)
print("Frames per second: ", fps) #프래임 속도 측정
frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
ret, img_ori = capture.read()
if not ret:
    print("Can't receive frame. Exiting ...")
    exit()

out = cv2.VideoWriter(r'C:\Users\DGSW\OneDrive\바탕 화면\Bird_eye_line_detect\videos\output_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

while True:
    ret, img_src = capture.read()
    if ret ==False:
        print("동영상 종료")
        break

    ######## perspective_transform.ipynb #############################
    calibration_data = pickle.load(open('C:\\Users\\DGSW\PycharmProjects\\notebooks\\calibration_data.p', "rb"))

    matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']

    source_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
    dest_points = [(320, 0), (320, 720), (960, 720), (960, 0)]
    birdsEye = BirdsEye(source_points, dest_points, matrix, dist_coef)
    ####------------------------------------------------------------
    img_src1 = img_src.copy()
    img_dst1 = birdsEye.undistort(img_src1, show_dotted=False)  # 외곡보정
    draw_line_n_mark_circle(img_dst1, source_points)  # 선그리기
    # cv2.imshow('undistorted', img_dst_1)

    ####------------------------------------------------------------
    img_dst2 = birdsEye.sky_view(img_src1, show_dotted=False)
    # cv2.imshow('sky_view', img_dst_2)

    #################################################################
    # gradient_and_color_thresholding.ipynb
    p = {'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
         'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}
    laneFilter = LaneFilter(p)  # lanefilter.py에서 laneFilter class의 객체를 선언

    img_src2 = img_src.copy()
    img_bird = birdsEye.sky_view(img_src2, show_dotted=False)
    img_dst2 = birdsEye.undistort(img_src2, show_dotted=False)
    binary = laneFilter.apply(img_dst2)

    ### masked_lane :boolean
    masked_lane = np.logical_and(birdsEye.sky_view(binary), roi(binary))
    masked_lane = masked_lane.astype(np.uint8) * 255
    masked_lane = cv2.cvtColor(masked_lane, cv2.COLOR_GRAY2BGR)

    sobel_img = birdsEye.sky_view(laneFilter.sobel_breakdown(img_dst2))
    sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_RGB2BGR)
    color_img = birdsEye.sky_view(laneFilter.color_breakdown(img_dst2))
    # print(img_dst2.shape)
    # result = merge_n_resize_4images(img_bird,color_img,sobel_img,masked_lane)
    # cv2.imshow('temp',result)

    #################################################################
    #### curve fitting #############################################
    curves = Curves(number_of_windows=9, margin=100, minimum_pixels=50,
                    ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)
    img_src3 = img_src.copy()
    binary = laneFilter.apply(img_src3)

    wb = np.logical_and(birdsEye.sky_view(binary), roi(binary)).astype(np.uint8)
    # cv2.imshow('wb',wb*255)
    result = curves.fit(wb)
    print("[real world] left best-fit curve parameters:", result['real_left_best_fit_curve'])
    print("[real world] right best-fit curve parameters:", result['real_right_best_fit_curve'])
    print("[pixel] left best-fit curve parameters:", result['pixel_left_best_fit_curve'])
    print("[pixel] left best-fit curve parameters:", result['pixel_right_best_fit_curve'])
    print("[left] current radius of curvature:", result['left_radius'], "m")
    print("[right] current radius of curvature:", result['right_radius'], "m")
    print("vehicle position:", result['vehicle_position_words'])

    # cv2.imshow('result',result['image'])
    ## 4영상 합쳐서 사이즈 줄이기
    img_result = merge_n_resize_4images(img_bird, color_img, sobel_img, result['image'])
    # cv2.imshow('4 merged images',img_result)
    #####################################################################
    ### 주행 도로 색깔 표시
    img_src4 = img_src.copy()
    # binary = laneFilter.apply(img_src4)
    # wb = np.logical_and(birdsEye.sky_view(binary), roi(binary)).astype(np.uint8)
    # result = curves.fit(wb)
    print("[real world] left best-fit curve parameters:", result['real_left_best_fit_curve'])
    print("[real world] right best-fit curve parameters:", result['real_right_best_fit_curve'])
    print("[pixel] left best-fit curve parameters:", result['pixel_left_best_fit_curve'])
    print("[pixel] left best-fit curve parameters:", result['pixel_right_best_fit_curve'])
    print("[left] current radius of curvature:", result['left_radius'], "m")
    print("[right] current radius of curvature:", result['right_radius'], "m")

    img_src4 = birdsEye.project(img_src4, binary, result['pixel_left_best_fit_curve'], result['pixel_right_best_fit_curve'])
    height, width = img_result.shape[:2]
    img_src4[0:height, 0:width, :] = img_result
    img_result = cv2.putText(img_src4, result['vehicle_position_words'], (580, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #pts = np.array([[int(width / 2), 720], [int(width / 2), int(height / 2)]], np.int32)
    cv2.line(img_result,(int(width / 2), 460), (int(width / 2), 720),(0, 0, 255), thickness = 2)
    numbers = result['vehicle_position_words'].split()
    numbers1 = numbers[0]
    numbers2 = numbers[2]
    numbers1 = float(numbers1)*20
    print(numbers1, numbers2)
    if numbers2 == 'right':
        cv2.line(img_result, (int(width / 2) + int(numbers1), 560), (int(width / 2) + int(numbers1), 600), (255, 0, 0), thickness=20)
    elif numbers2 == 'left':
        cv2.line(img_result, (int(width / 2) - int(numbers1), 560), (int(width / 2) - int(numbers1), 600), (255, 0, 0), thickness=20)
    out.write(img_result)
    cv2.imshow('result', img_result)
    key = cv2.waitKey(30)
    if key == 27:
        break
#######################################################################
capture.release()
cv2.destroyAllWindows()
