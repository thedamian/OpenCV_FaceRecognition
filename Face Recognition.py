import face_recognition
import cv2
import numpy as np

# Get reference to the defaul webcam #0
video_capture = cv2.VideoCapture(0)

# load a sample photo to learn facial features
todd_image = face_recognition.load_image_file('todd.jpeg')
todd_face_encoding = face_recognition.face_encodings(todd_image)[0]

# let's train a second face
damian_image = face_recognition.load_image_file('damian.jpg')
damian_face_encoding = face_recognition.face_encodings(damian_image)[0]

# create a list of known face encodings and their corresponding names
known_face_encodings = [todd_face_encoding, damian_face_encoding]
known_face_names = ['Todd A.', 'Damian M.']

# initialize a few variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
  # grab a single frame of video
  ret, frame = video_capture.read()
  # to save time, only process every other frame
  if process_this_frame:
    # process frame
    # resize the video to 1/4 size for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # convert from BGR color (OpenCV uses) to RGB (face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # find all faces and encodings in current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
      # see if there is a match
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
      name = "Unknown"
      # if a match was found... use the face with the smallest distance to the new face
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      
      face_names.append(name)

  process_this_frame = not process_this_frame

  # now we need to draw each red box with the name underneath
  for (top, right, bottom, left), name in zip(face_locations, face_names):
    # scale back to full frame
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    # draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # draw the label with the name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

  # display the resulting frame
  cv2.imshow('Face Detection - hit q to quit', frame)

  # hit 'q' to quit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()