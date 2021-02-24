import enum
import numpy as np
import tflite_runtime.interpreter as tflite

class BodyPart(enum.Enum):
  NOSE = 0
  LEFT_EYE = 1
  RIGHT_EYE= 2
  LEFT_EAR= 3
  RIGHT_EAR= 4
  LEFT_SHOULDER= 5
  RIGHT_SHOULDER = 6
  LEFT_ELBOW = 7
  RIGHT_ELBOW = 8
  LEFT_WRIST= 9
  RIGHT_WRIST= 10
  LEFT_HIP= 11
  RIGHT_HIP= 12
  LEFT_KNEE= 13
  RIGHT_KNEE = 14
  LEFT_ANKLE = 15
  RIGHT_ANKLE = 16

class Position:
    def __init__(self, x=0,y=0):
        self.x = x
        self.y = y

class KeyPoint:
    def __init__(self,bodypart = BodyPart.NOSE, position = Position() , score=0.0 ):
        self.bodyPart = bodypart
        self.position = position
        self.score = score

class Person:
    def __init__(self,keypoints = [] , score=0.0):
        self.keyPoints = keypoints
        self.score = score

class Posenet:

    def __init__(self,context , model_path="posenet_model.tflite"):
        self.lastInferenceTimeNanos = -1
        self.interpreter = None
        self.gpuDelegate = None
        self.model_path = model_path
        self.context = context
        self.NUM_LITE_THREADS  = 4


    def getInterpreter(self):
        if self.interpreter is not None:
            return self.interpreter
        interpreter = tflite.Interpreter(model_path=self.model_path , num_threads = self.NUM_LITE_THREADS)
        interpreter.allocate_tensors()
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.interpreter = interpreter
        return interpreter

    def close(self):
        self.interpreter.close()
        self.interpreter = None

    def sigmoid(self , x):
        return (1 / (1 + np.exp(-x)))

    def getKeyPointLocations(self, heatmaps):
        height , width , numKeyPoints = heatmaps.shape
        keypointPositions = np.zeros(numKeyPoints)
        for keypoint  in range(numKeyPoints):
            maxVal  = heatmaps[0][0][keypoint ]
            maxRow  , maxCol = 0
            for row in range(height):
                for col in range(width):
                     if (heatmaps[row][col][keypoint] > maxVal):
                         maxVal = heatmaps[row][col][keypoint]
                         maxRow = row
                         maxCol = col

            keypointPositions[keypoint] = (maxRow, maxCol)

        return keypointPositions

    def confidenceScores(self,heatmaps ,offsets,keypointPositions , height , width, HEIGHT , WIDTH):
        numKeyPoints = len(keypointPositions)
        xCoords = np.zeros(numKeyPoints)
        yCoords = np.zeros(numKeyPoints)
        confidenceScores  = np.zeros(numKeyPoints)

        for idx ,position in enumerate(keypointPositions):
            positionY  = keypointPositions[idx][0]
            positionX = keypointPositions[idx][1]
            yCoords[idx] = int( position[0] / float(height - 1) * HEIGHT + offsets[positionY][positionX][idx])
            xCoords[idx] = int( position[1] / float(width - 1) * WIDTH + offsets[positionY][positionX][idx])
            confidenceScores[idx] = self.sigmoid(heatmaps[positionY][positionX][idx])

        return xCoords , yCoords , confidenceScores

    def getPersonDetails(self , numKeyPoints , xCoords , yCoords,confidenceScores):
        person = Person()
        keypointList = [KeyPoint()]*numKeyPoints
        totalScore = 0
        for idx,it in enumerate(BodyPart):
            keypointList[idx].bodyPart = it
            keypointList[idx].position.x = xCoords[idx]
            keypointList[idx].position.y = yCoords[idx]
            keypointList[idx].score  = confidenceScores[idx]
            totalScore += confidenceScores[idx]

        person.keyPoints = keypointList
        person.score = totalScore / numKeyPoints
        return person

    def estimateSinglePose(self, image):
        HEIGHT, WIDTH  = self.input_details[0]["shape"][1:3]
        input_data = np.expand_dims(image.resize((WIDTH ,HEIGHT)), axis=0)
        input_mean , input_std = 127.5  ,127.5
        input_data = (np.float32(input_data) - input_mean) / input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        heatmaps  = self.interpreter.get_tensor(self.output_details[0]['index'])
        heatmaps  = np.squeeze(heatmaps)

        offsets   = self.interpreter.get_tensor(self.output_details[1]['index'])
        offsets   = np.squeeze(offsets )

        height , width , numKeyPoints = heatmaps.shape

        keypointPositions = self.getKeyPointLocations(heatmaps , offsets)

        xCoords , yCoords , confidenceScores = (self.getConfidenceScores(heatmaps
                                                    , offsets
                                                    ,keypointPositions
                                                    , height
                                                    , width
                                                    , HEIGHT
                                                    , WIDTH))

        person = self.getPersonDetails( numKeyPoints , xCoords , yCoords,confidenceScores)
        return person
