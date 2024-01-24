import os
import uuid

# this script changes filenames to give them unique id's
# folder structure should look like this:
#
# ./uuidify_frames.py
# ./dataset/labels/
# ./dataset/images/

def main():

    dir = os.getcwd() + "/dataset-with-non-unique-filenames"
    labels_dir = dir + "/labels"
    images_dir = dir + "/images"

    counter = 0
    
    for filename in sorted(os.listdir(labels_dir)):
        name, _ = os.path.splitext(filename)

        if filename.endswith('.txt'):
            
            # generate uuid
            old_name, _ = os.path.splitext(filename)
            uuid_str = str(uuid.uuid4())
            new_name = old_name + "-" + uuid_str
            
            # rename label file
            os.rename(labels_dir + "/" + old_name + ".txt", labels_dir + "/" + new_name + ".txt")
            print("mv " + filename + " " + new_name + ".txt")
            
            # rename image file
            os.rename(images_dir + "/" +  old_name + ".jpg", images_dir + "/" + new_name + ".jpg")
            print("mv " + old_name + ".jpg" + " " + new_name + ".jpg")

            counter += 1

    print("files: " + str(counter))

if __name__ == "__main__":
    main()
