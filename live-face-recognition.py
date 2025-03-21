import face_recognition 
import cv2
import numpy as np

# get reference to the default webcam
video_capture = cv2.VideoCapture(0)

# load a sample photo to learn facial fature
damian_image = face_recognition.load_image_file('damian.jpg')
damian_face_encoding = face_recognition.face_encodings(damian_image)[0]

# let's train a second face
todd_image = face_recognition.load_image_file("David1.png")
todd_face_encoding = face_recognition.face_encodings(todd_image)[0]

# create a list of known face encodings and coresponding names
known_face_encodings = [damian_face_encoding, todd_face_encoding]
known_face_names = ['Damian', 'David']

print(damian_face_encoding)
print("----------------------")
print(todd_face_encoding)

#initialize a few variables
face_locations=[]
face_encodings=[]
face_names = []
processing_this_frame = True

while True:
    # grab single frace of video
    ret,frame = video_capture.read()
    # to save time only process every other frame
    if processing_this_frame == True:
        #process frame
        small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
        #convert the RGB color(OPenCV uses to RGB (facerecognition))
        rgb_msall_frame = small_frame[:,:,::-1]

        # find all face in current frame of video
        face_locations = face_recognition.face_locations(rgb_msall_frame)
        face_encodings = face_recognition.face_encodings(rgb_msall_frame, face_locations)

        face_names = []
        for face in face_encodings:
            # see if there's a match
            matches = face_recognition.compare_faces(known_face_encodings, face)
            name = "Unknown"
            # if a match was found just use the face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
    # draw the red box and the name
    for (top,right,bottom,left),name in zip(face_locations, face_names):
        #scale these back to full frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #draw a box around the face
        cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255),2 )
        # draw the label with the name
        cv2.rectangle(frame, (left,bottom-35), (right,bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



    processing_this_frame = not processing_this_frame
    cv2.imshow("Face Detection - Hit q to quit", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()