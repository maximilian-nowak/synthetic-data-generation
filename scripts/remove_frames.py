import os

# this script removes frames from a dataset
# tweak the modulo operation for more or less frames
# folder structure should look like this:
#
# ./remove_frames.py
# ./dataset/labels/
# ./dataset/images/

def main():

    dir = os.getcwd() + "/first_trainings_set_norm"
    labels_dir = dir + "/labels"
    images_dir = dir + "/images"

    counter = 0
    del_counter = 0
    
    for filename in sorted(os.listdir(labels_dir)):
        
        if filename.endswith('.txt'):
            if not counter % 2 == 0:
                name, _ = os.path.splitext(filename)
                
                # remove label file
                os.remove(labels_dir + "/" + name + ".txt")
                print("rm " + labels_dir + "/" + name + ".txt")
                
                # remove image file
                os.remove(images_dir + "/" + name + ".jpg")
                print("rm " + images_dir + "/" + name + ".jpg")
            
                del_counter += 1
        
            counter += 1

    print("remaining frames: " + str(counter - del_counter))
    print("deleted frames: " + str(del_counter))

if __name__ == "__main__":
    main()
