# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,    help="path to input video file")
args = vars(ap.parse_args())

# initialise detector
ARUCO_DICT = {"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT['DICT_APRILTAG_36h11'])
arucoParams = cv2.aruco.DetectorParameters_create()

# initialise tracker
OPENCV_OBJECT_TRACKERS = {'kcf': cv2.TrackerMIL_create}
tracker = OPENCV_OBJECT_TRACKERS['kcf']()

# bounding box coords
currInitBBs = None
success = None
firstTrack = False

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# loop over the frames from the video stream
while True:

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 1000 pixels

    frame = imutils.resize(frame, width=750)
    (H, W) = frame.shape[:2]
    # detect ArUco markers in the input frame
    (currCorner, currIds, rejected) = cv2.aruco.detectMarkers(frame,
        arucoDict, parameters=arucoParams)

    # verify *at least* one ArUco marker was detected
    if len(currCorner) > 0:
        # flatten the ArUco currIds list
        currIds = currIds.flatten()
        currInitBBs = []
        # loop over the detected ArUCo currCorner
        for (markerCorner, markerID) in zip(currCorner, currIds):
            # extract the marker currCorner (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)

            currCorner = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = currCorner
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            cv2.circle(frame, topRight, 4, (0, 0, 255), 10)

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (255, 0, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 255), 2)
            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x = topLeft[0]
            y = topLeft[1]
            w = topRight[0] -x
            h = bottomLeft[1] -y

            currInitBBs.append((x,y,w,h))
            
        prevInitBBs = currInitBBs
        prevCorner = currCorner
        prevIds = currIds
        firstTrack = True

    elif (len(currCorner) == 0) and firstTrack:
        for prevInitBB in prevInitBBs:
            tracker.init(frame, prevInitBB)
        firstTrack = False

    # NEED TO UPDATE THE INITBB FOR THE TRACK TO FOLLOW
    if (len(currCorner) == 0) and (currInitBBs is not None) and (not firstTrack):
        currInitBBs = []
        for prevInitBB in prevInitBBs:
            tracker.init(frame, prevInitBB)
            # fps = FPS().start()
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                currInitBBs.append((x,y,w,h))
        
        prevInitBB = currInitBBs

            # update the FPS counter
            # fps.update()
            # fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame

    # log
    info = [
        ('Detector','DICT_APRILTAG_36h11') if len(currCorner) > 0 else ("Tracker", 'kcf'),
        ("Tracking", "Yes" if success else "No"),
        # ("FPS", "{:.2f}".format(fps.fps())),
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()