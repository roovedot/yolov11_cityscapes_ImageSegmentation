# Custom Image Segmentation for Autonomous Driving Aplications

link to Notes/Documentation -> https://massive-birch-56b.notion.site/Custom-Yolov8-Object-Detection-Model-for-Autonomous-Driving-Applications-141d04bb941780a0ae4df644c2ea04f1

This is the **second and final version** of a custom Object detection Model aimed at **recognising cars, bicicles, pedestrians, and other obstacles which may appear on the road through a dashcam.**

Following my **[V1: Custom Yolov8 Object Detection Model for Autonomous Driving Applications](https://www.notion.so/V1-Custom-Yolov8-Object-Detection-Model-for-Autonomous-Driving-Applications-141d04bb941780a0ae4df644c2ea04f1?pvs=21)** Proyect, this time I took a different approach, instead of bounding boxes I am going for **Image Segmentation**, which will provide a more complete and precise understanding of the environment.

The objective is to **implement this model to assist, in real time, a Self-Driving Car** in making precise and more informed decisions.

This time, I have acces to a **MUCH bigger dataset** (cityscapes Dataset) and a **more powerful machine** to be able to train the model, kindly provided by my University [(Universidad Europea de Madrid)](https://universidadeuropea.com/conocenos/madrid/)

To make this model, I fine-tuned ultralytics **Yolov11n-seg.pt** Model on [**CityScapes Dataset](https://www.cityscapes-dataset.com/),** for instance Image Segmentation.