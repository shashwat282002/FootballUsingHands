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

