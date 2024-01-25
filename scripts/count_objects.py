import os

# this script counts class occurrences of each traffic object
# useful for documenting new datasets
# folder structure should look like this:
#
# ./remove_frames.py
# ./dataset/labels/

names = ['trafficlight_green','trafficlight_red', 'pit_in','pit_out', 'park_parallel','park_cross', 'overtaking_prohibited', 'overtaking_permitted']
occurrences = [0,0,0,0,0,0,0,0]
    
def countLabels(input_filename):
    with open(input_filename, 'r') as input_file:
        for line in input_file.readlines():
            class_label = line.split()[0]
            occurrences[int(class_label)] += 1

def main():
    # dataset = [('adc_4_haube_0_uuidified', './adc_4_haube_0_uuidified/labels')]
    # dataset = [('testfolder', './testfolder/labels')]
    # dataset = [('synthetic_traffic_signs_uudified', './synthetic_traffic_signs_uudified/labels')]
    dataset = [('synthetic_traffic_mixed_uudified', './synthetic_traffic_mixed_uudified/labels')]
    
    
    total_counter = 0
    for parent, input_folder in dataset:
        counter = 0
        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith('.txt'):
                countLabels(os.path.join(input_folder, filename))
                counter += 1
                    
        total_counter += counter
        print(parent)
        
    print("frames: " + str(total_counter))
    print("objects: " +str(list(zip(names, occurrences))))

if __name__ == "__main__":
    main()
