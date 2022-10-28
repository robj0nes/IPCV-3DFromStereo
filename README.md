# Lab5-3D-from-stereo
Python code and lab sheet for lab class 5 on stereo. Details on how to run the code are below and the lab tasks can be found in the lab sheet IPCV-3DLab1-22-23.pdf.

<ol>
  <li> Down load a copy of LabI-v1.py or LabI-v2.py. Both have the same functionality but may be OS dependent. Try v1 first and then v2. v2 recommended for MacOS. If neither work, contact a TA.

  <li> Install a virtual environment using conda: <tt> conda create -n ipcv python=3.8</tt>

  <li> Activate the virtual environment: <tt> conda activate ipcv</tt>

  <li> Install opencv: <tt> pip install opencv-python</tt> or <tt> conda install -c menpo opencv </tt>

  <li> Install open3d: <tt> pip install open3d</tt> or <tt> conda install -c open3d-admin open3d</tt>

  <li> Run the simulator: <tt> python LabI-v1.py</tt> (if error, then try <tt> python LabI-v2.py</tt>)
  </ol>


## Troubleshooting

On Mac, you might need to install LLVM's OpenMP runtime library `brew install libomp`
