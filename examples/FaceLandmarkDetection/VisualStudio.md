# For clueless Visual Studio users

[![DOCS](https://user-images.githubusercontent.com/7437173/138705052-1112c657-6830-455e-b8d8-2307d1960ab6.PNG)](https://youtu.be/cDA3_5982h8)

Not sure whether this is a good way to write docs, but it's definitely funny.

Assuming you've generated Visual Studio solution files:

1. Find `FabSoften.sln` in the `build folder` -> <kbd>double-click</kbd> `FabSoften.sln` to open the solution file in `Visual Studio`,
2. Find `Solution Configurations`(defaults to `Debug` `x64`) and change `Debug` to `Release`:   
  - <kbd>left-click</kbd> `Debug` to open a dropdown menu 
  - <kbd>left-click</kbd> `Release` in the menu
3. Find the project named `FaceLandmarkDetection` in the `Solution Explorer`:
  - <kbd>right-click</kbd> `FaceLandmarkDetection`
  - <kbd>left-click</kbd> `Build` in the prompt menu
  - <kbd>right-click</kbd> `FaceLandmarkDetection` again
  - <kbd>left-click</kbd> `Open in Terminal` in the prompt menu
4. Copy the code in the following code block to the `Develop Powershell`: 
```
.\Release\FaceLandmarkDetection.exe ..\..\..\assets\pexels-aadil-2598024.jpg ..\..\..\models\shape_predictor_68_face_landmarks.dat
```
  - Hover your cursor onto the above code block
  - Now a button appears at the right hand side of the code block
  - Move your cursor over that button and <kbd>left-click</kbd> the button
5. Find the `Develop Powershell` window and hover your cursor on the window, then paste the code by doing a <kbd>right-click</kbd>,
6. Press <kbd>enter</kbd> and an image will show up with detected landmarks,
7. To exit, press <kbd>Esc</kbd>.



