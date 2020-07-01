import cv2, time, string, copy, os, sys, getopt
import numpy as np
from os import listdir

def main(argv):
    global keypoint_dict
    # Global values
    keypoint_dict = {}
    quitted = False
    read_file = "./hands"
    watch_file = ""
    start_at = 0
    file_list = []
    
    # Parsing CLI arguments
    try:
        if argv.index('--begin') < len(argv) - 1:
            spot = argv.index('--begin')
            if argv[spot + 1].isnumeric():
                start_at = int(argv[spot + 1])
            else:
                quitted = True
                watch_file = argv[spot + 1]
        if argv.index('--read') < len(argv) - 1:
            spot = argv.index('--read')
            read_file = argv[spot + 1]
    except:
        print('-')
        # sys.exit(1)

    try:
        file_list = listdir(read_file)
        file_list.sort()
    except:
        print('-')


    # Beginning to read into each image in folder
    for num in range(len(file_list)):
        # Global values for file scope
        global cache, img, change_count, count_cache, keypoints
        fl = file_list[num]
        cache = []
        change_count = 1
        count_cache = []
        if watch_file != "" and watch_file in str(fl):
            quitted = False


        if not quitted and num > start_at - 1:
            filename = f"{read_file}/{str(fl)}"
            protoFile = "hand/pose_deploy.prototxt"
            weightsFile = "hand/pose_iter_102000.caffemodel"
            POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            nPoints = 22

            img = cv2.imread(filename)
            scale = 600 / img.shape[0]
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            dim = (width, height)
            img = cv2.resize(img, dim)
            frameCopy = np.copy(img)
            frameWidth = img.shape[1]
            frameHeight = img.shape[0]
            aspect_ratio = frameWidth/frameHeight

            threshold = 0.1
            t = time.time()
            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio*inHeight)*8)//8)
            # Center coordinates 
            center_coordinates = (369, 303) 
            # Radius of circle 
            radius = 5
            # Blue color in BGR 
            color = (255, 0, 0) 
            # Line thickness of 2 px 
            thickness = -1

            # DNN CV2 HAND DETECTION MODEL
            inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            print("time taken by network : {:.3f}".format(time.time() - t))

            keypoints = []

            def click_and_crop(event, x, y, flags, param):
                # grab references to the global variables
                global keypoints, keypoint_dict, cache, img, change_count, count_cache
                
                # if the left mouse button was clicked, record the starting
                # (x, y) coordinates and indicate that cropping is being
                # performed
                if event == cv2.EVENT_LBUTTONDOWN and len(keypoints) < 21:
                    refPt = [x, y]
                    keypoints.append(refPt)
                    print(f"You just posted for keypoint #{len(keypoints)} for file {str(fl)}")
                    print("==== Press 'n' to continue to the next image")
                    print("==== Press 'b' to go back a keypoint")
                    print("==== Press 'q' to quit")
                    keypoint_dict[str(fl)] = keypoints

                    
                    length = len(keypoints)
                    if (length > 1):
                        if length == 6 or length == 10 or length == 14 or length == 18:
                            cv2.line(img, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[-1][0]), int(keypoints[-1][1])), (255, 0, 0), 2)
                        else:
                            cv2.line(img, (int(keypoints[-2][0]), int(keypoints[-2][1])), (int(keypoints[-1][0]), int(keypoints[-1][1])), (255, 0, 0), 2)
                    cv2.circle(img, (int(keypoints[-1][0]), int(keypoints[-1][1])), 5, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.imshow('Output-Skeleton', img)

                    cached_image = copy.deepcopy(img)
                    cache.append(cached_image)

                    count_cache.append(change_count)
                    change_count += 1


                    if len(keypoints) >= 21:
                        print("=======================================================")
                        print("=======================================================")
                        print("DONE WITH", len(keypoint_dict), "IMAGES SO FAR!!")
                        print("==== Press 'n' to continue to the next image")
                        print("==== Press 'b' to go back a keypoint")
                        print("==== Press 'q' to quit")
                        print("=======================================================")
                        print("=======================================================")


            points = []

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))
                
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                
                if prob > threshold :
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                
                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                else :
                    points.append(None)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(img, points[partA], points[partB], (255, 255, 255), 1)
                    cv2.circle(img, points[partA], 1, (255, 255, 255, 0.4), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(img, points[partB], 1, (255, 255, 255, 0.4), thickness=-1, lineType=cv2.FILLED)
    
            cv2.namedWindow('Output-Skeleton')
            cv2.setMouseCallback('Output-Skeleton', click_and_crop)
            cv2.imshow('Output-Skeleton', img)
            
            while True:
                k = cv2.waitKey(0)
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    quitted = True
                    break
                if k == ord('b'):
                    if len(cache) > 1:
                        keypoints.pop()
                        cache = cache[:len(cache) - 1]
                        count_cache.pop()
                        
                        img = cache[-1]
                if k == ord('n') or len(keypoints) >= 21:
                    break

                cv2.imshow('Output-Skeleton', img)
                
            cv2.destroyAllWindows()
            cache = []


    if len(keypoint_dict) > 0:
        file_count = 1
        keypoint_file = ""
        file_available = not os.path.isfile('keypoint_labels.json')
        if file_available:
            keypoint_file = 'keypoint_labels.json'
        while not file_available:
            check_file = os.path.isfile(f"keypoint_labels_{file_count}.json")
            if not check_file:
                keypoint_file = f"keypoint_labels_{file_count}.json"
                break
            file_count += 1

        outF = open(keypoint_file, "w")
        outF.write("{\n")
        for idx, (key, val) in enumerate(keypoint_dict.items()):
            if idx == 0:
                outF.write(f"\t\"{key}\": [")
                if len(val) > 0:
                    for i in range(len(val)):
                        if i == 0:
                            outF.write(f"\n\t\t{val[i]}")
                        else:
                            outF.write(f",\n\t\t{val[i]}")
                outF.write(f"\n\t]")
            else:
                outF.write(f",\n\t\"{key}\": [")
                if len(val) > 0:
                    for i in range(len(val)):
                        if i == 0:
                            outF.write(f"\n\t\t{val[i]}")
                        else:
                            outF.write(f",\n\t\t{val[i]}")
                outF.write(f"\n\t]")
        outF.write("\n}")
        outF.close()


if __name__ == "__main__":
   main(sys.argv[1:])
