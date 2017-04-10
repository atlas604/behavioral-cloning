with open('Users/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2            # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        directory = 'Users/IMG/'    # fill in the path to your training IMG directory
        img_center = process_image(np.asarray(Image.open(path + row[0])))
        img_left = process_image(np.asarray(Image.open(path + row[1])))
        img_right = process_image(np.asarray(Image.open(path + row[2])))

        # add images and angles to data set
        car_images.extend(img_center, img_left, img_right)
        steering_angles.extend(steering_center, steering_left, steering_right)
