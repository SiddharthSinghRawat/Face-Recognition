import cv2
import face_recognition

# Function to capture faces and recognize them
def recognize_faces():
    # Open a connection to the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Load a sample image and learn how to recognize it
    sample_image = face_recognition.load_image_file("elon.jpg")
    sample_encoding = face_recognition.face_encodings(sample_image)[0]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame)
        
        # Recognize faces
        for face_location in face_locations:
            # Extract the face encoding for the current face
            face_encoding = face_recognition.face_encodings(frame, [face_location])[0]

            # Compare the current face encoding with the sample face encoding
            results = face_recognition.compare_faces([sample_encoding], face_encoding)

            name = "Unknown"
            if results[0]:
                name = "Registered Face"

            # Draw rectangle and label around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the face recognition function
recognize_faces()
