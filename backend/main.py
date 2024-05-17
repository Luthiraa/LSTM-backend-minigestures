import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import time 
import mediapipe as mp
import sklearn.model_selection as trainTestSplit
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# load the mediapipe pose model
mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

def mpdetection(image, model):
    #convert the image to rgsd
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    #detect the pose and make predictions
    results = model.process(image)
    image.flags.writeable = True
    #convert that jawn back to bgr
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def drawPose(image, results):
    #define drawing specifications
    landmarkDrawingSpec = mp.solutions.drawing_utils.DrawingSpec(color=(223, 27, 27), thickness=2, circle_radius=2)
    connectionDrawingSpec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

    #draw landmarks and connections
    mp.solutions.drawing_utils.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp.solutions.holistic.HAND_CONNECTIONS, 
        landmarkDrawingSpec, 
        connectionDrawingSpec
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp.solutions.holistic.HAND_CONNECTIONS, 
        landmarkDrawingSpec, 
        connectionDrawingSpec
    )
    return image

capture = cv2.VideoCapture(0)

with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        ret, frame = capture.read()
        #detect the pose
        image, results = mpdetection(frame, holistic)
        #draw the pose
        drawPose(image, results)

        if ret:
            cv2.imshow('frame', image)
            #leave the camera
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print("Failed to grab frame")
            break

    capture.release()
    cv2.destroyAllWindows()


def extractKeypoints(results):
    #if results are detected, extract the landmarks
    if results.right_hand_landmarks:
        rightHand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    #otherwise, create zeros array of 21x3 
    else:
        rightHand = np.zeros(21*3)

    if results.left_hand_landmarks:
        leftHand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        leftHand = np.zeros(21*3)

    return np.concatenate([rightHand, leftHand])



dataPath = os.path.join('data')
#actions
actions = np.array(['hello', 'rock', 'swipe_left', 'swipe_right', 'stop'])
#30 videos of each action/frame sets
numberSequence = 30
#30 frames in length per action respectively 
sequenceLength = 30 

for action in actions:
    for sequence in range(numberSequence):
        try:
            os.makedirs(os.path.join(dataPath, action, str(sequence)))
        except:
            pass

with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(numberSequence):
            for frameNum in range(sequenceLength):
                #read the frame
                ret, frame = capture.read()
                #detect the pose
                image, results = mpdetection(frame, holistic)
                #draw the pose
                drawPose(image, results)
                #save the frame
                if frameNum == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} video number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} video number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


                keypoints = extractKeypoints(results)
                npyPath = os.path.join(dataPath, action, str(sequence), str(frameNum))
                np.save(npyPath, keypoints)
                #display the frame
                cv2.imshow('frame', image)
                #leave the camera
              
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    capture.release()
    cv2.destroyAllWindows()

labelMap = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(numberSequence):
        window = []
        for frameNum in range(sequenceLength):
            res = np.load(os.path.join(dataPath, action, str(sequence), '{}.npy'.format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[action])

x = np.array(sequences)
print(x.shape)
y = to_categorical(np.array(labels)).astype(int)

#split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = trainTestSplit.train_test_split(x, y, test_size=0.05)

logDir = os.path.join('Logs')
tbCallback = TensorBoard(log_dir=logDir)

model=Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(xTrain, yTrain, epochs=2000, callbacks=[tbCallback])

model.save('action.h5')