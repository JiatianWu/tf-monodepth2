- Install libraries:
  - Make sure your python version is 3.7

   - If you use kinect, install Kinect related lib, unplug and plug kinect after setting up udev rules
    ````
    curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
    sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
    sudo apt-get update

    sudo apt install k4a-tools
    sudo apt install libk4a1.3-dev

    sudo cp kinect_tools/99-k4a.rules /etc/udev/rules.d/
    ````

  - Install open3d

   ````
    pip3 install open3d
   ````


- How to record data:

  - Make sure the kinect is charged:

   ````
      python3 kinect_data_recorder.py  --output filename_you_want_to_save.mkv
   ```` 
  - Click the Gui, press space if you want to start recording. When you want to stop recording,
      press space and then press esc.