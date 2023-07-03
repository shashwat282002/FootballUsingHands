#
# import mediapipe as mp
# import cv2
# import cvzone
# import numpy as np
# from cvzone.HandTrackingModule import HandDetector
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon=0.8, maxHands=2)
#
# cap.set(3, 1280)
# cap.set(4, 720)
# ImageBackground = cv2.imread("PongGame/footballfield.jpeg")
# Football = cv2.imread('PongGame/newfootball.png', cv2.IMREAD_UNCHANGED)
# Lplayer = cv2.imread('PongGame/playerleft.png', cv2.IMREAD_UNCHANGED)
# Rplayer = cv2.imread('PongGame/player right.png', cv2.IMREAD_UNCHANGED)
# GameOver = cv2.imread('PongGame/GameOver.png')
#
# ballpos = [610, 340]
# speedx = 20
# speedy = 20
# gameover = False
# score_left = 0  # Score for the left player
# score_right = 0  # Score for the right player
#
# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#
#     img = cv2.addWeighted(img, 0.2, cv2.resize(ImageBackground, (1280, 720)), 0.8, 0)
#
#     if hands:
#         for hand in hands:
#             x, y, w, h = hand['bbox']
#             h1, w1 = 100, 100
#             y1 = y - h1 // 2
#             y1 = np.clip(y1, 90, 505)
#
#             if hand['type'] == "Left":
#                 img = cvzone.overlayPNG(img, cv2.resize(Lplayer, (100, 100)), (30, y1))
#                 if 30 < ballpos[0] < 30 + w1 and y1 - 5 < ballpos[1] < y1 + h1 + 5:
#                     speedx = -speedx
#                     ballpos[0] += 30
#
#                 # Left player misses the ball
#                 if ballpos[0] > 1160:
#                     score_left += 1  # Increase the score for the right player
#                     ballpos = [610, 340]  # Reset the ball position
#                     if score_left == 1:  # Check if the right player wins
#                         gameover = True
#
#             if hand['type'] == "Right":
#                 img = cvzone.overlayPNG(img, cv2.resize(Rplayer, (100, 100)), (1140, y1))
#                 if 1140 - 50 < ballpos[0] < 1140 + w1 and y1 - 5 < ballpos[1] < y1 + h1 + 5:
#                     speedx = -speedx
#                     ballpos[0] -= 30
#
#                 # Right player misses the ball
#                 if ballpos[0] < 30:
#                     score_right += 1  # Increase the score for the left player
#                     ballpos = [610, 340]  # Reset the ball position
#                     if score_right == 1:  # Check if the left player wins
#                         gameover = True
#
#     img = cvzone.overlayPNG(img, cv2.resize(Football, (50, 50)), ballpos)
#
#     if gameover:
#         img = cv2.addWeighted(img, 0.2, cv2.resize(GameOver, (1280, 720)), 0.1, 0)
#         if score_right == 1:
#             cv2.putText(img, "Right Player Wins!", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
#         else:
#             cv2.putText(img, "Left Player Wins!", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
#     else:
#         if ballpos[1] >= 555 or ballpos[1] <= 115:
#             speedy = -speedy
#
#         ballpos[0] += speedx
#         ballpos[1] += speedy
#
#     # Display the scores for each player
#     cv2.putText(img, f"Left Player Score: {score_left}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(img, f"Right Player Score: {score_right}", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == ord('r'):
#         ballpos = [610, 340]
#         score_left = 0  # Reset the score for the left player
#         score_right = 0  # Reset the score for the right player
#         gameover = False
#
#     if (key & 0xFF == ord('x')):
#         break
#
# cv2.destroyAllWindows()

import mediapipe as mp
import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

cap.set(3, 1280)
cap.set(4, 720)
ImageBackground = cv2.resize(cv2.imread("PongGame/footballfield.jpeg"), (1280, 720))
Football = cv2.resize(cv2.imread('PongGame/newfootball.png', cv2.IMREAD_UNCHANGED), (50, 50))
Lplayer = cv2.resize(cv2.imread('PongGame/playerleft.png', cv2.IMREAD_UNCHANGED), (100, 100))
Rplayer = cv2.resize(cv2.imread('PongGame/player right.png', cv2.IMREAD_UNCHANGED), (100, 100))
GameOver = cv2.imread('PongGame/GameOver.png')

# Remove the alpha channel from the Football, Lplayer, and Rplayer images
Football = Football[:, :, :3]
Lplayer = Lplayer[:, :, :3]
Rplayer = Rplayer[:, :, :3]

ballpos = [610, 340]
speedx = 20
speedy = 20
gameover = False
score_left = 0  # Score for the left player
score_right = 0  # Score for the right player

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    img = cv2.addWeighted(img, 0.2, ImageBackground, 0.8, 0)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1 = 100, 100
            y1 = y - h1 // 2
            y1 = np.clip(y1, 90, 505)

            if hand['type'] == "Left":
                img[y1:y1 + h1, 30:30 + w1] = Lplayer
                if 30 < ballpos[0] < 30 + w1 and y1 - 5 < ballpos[1] < y1 + h1 + 5:
                    speedx = -speedx
                    ballpos[0] += 30

                # Left player misses the ball
                if ballpos[0] > 1160:
                    score_right += 1  # Increase the score for the right player
                    ballpos = [610, 340]  # Reset the ball position
                    if score_right == 5:  # Check if the right player wins
                        gameover = True

            if hand['type'] == "Right":
                img[y1:y1 + h1, 1140:1140 + w1] = Rplayer
                if 1140 - 50 < ballpos[0] < 1140 + w1 and y1 - 5 < ballpos[1] < y1 + h1 + 5:
                    speedx = -speedx
                    ballpos[0] -= 30

                # Right player misses the ball
                if ballpos[0] < 30:
                    score_left += 1  # Increase the score for the left player
                    ballpos = [610, 340]  # Reset the ball position
                    if score_left == 5:  # Check if the left player wins
                        gameover = True

    img[ballpos[1]:ballpos[1] + 50, ballpos[0]:ballpos[0] + 50] = Football

    if gameover:
        img = GameOver
        if score_left == 5:
            cv2.putText(img, "Left Player Wins!", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Right Player Wins!", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if ballpos[1] >= 555 or ballpos[1] <= 115:
            speedy = -speedy

        ballpos[0] += speedx
        ballpos[1] += speedy

    cv2.putText(img, f"Left Player Score: {score_left}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Right Player Score: {score_right}", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballpos = [610, 340]
        score_left = 0
        score_right = 0
        gameover = False

    if key & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Initialize OpenCV VideoCapture
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Read frame from webcam
#     success, frame = cap.read()
#     if not success:
#         break
#
#     # Convert the BGR image to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame with MediaPipe Hands
#     results = hands.process(rgb_frame)
#
#     # Check for hand landmarks
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Loop through all landmarks and draw them on the frame
#             for landmark in hand_landmarks.landmark:
#                 # Convert landmark coordinates from normalized values to pixel values
#                 height, width, _ = frame.shape
#                 x = int(landmark.x * width)
#                 y = int(landmark.y * height)
#                 cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
#
#     # Display the frame
#     cv2.imshow('Hand Tracking', frame)
#
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the VideoCapture and close the OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
