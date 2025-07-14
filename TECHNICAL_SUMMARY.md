
# ANPR System - Project Report Summary

## 1. Introduction & Project Goal

This document provides a comprehensive summary of the Automatic Number Plate Recognition (ANPR) project. The primary goal of this project was to develop a complete, end-to-end software application capable of automatically detecting, recognizing, and logging Nigerian vehicle license plates from digital images.

The system is designed as a full-stack web application. It provides a user-friendly interface for interaction, data visualization, and system administration, all powered by a sophisticated backend that leverages multiple machine learning models. The outcome is a powerful and practical tool for automated vehicle monitoring and data collection.

## 2. System Components

The project is built from several core components that work together to deliver its functionality. At a high level, these are:

*   **The ANPR Engine:** This is the "brain" of the system, responsible for all the intelligent analysis. When it receives an image, it uses a pipeline of Artificial Intelligence (AI) models to find and extract information about the vehicles and license plates within it.

*   **The Web Application:** This is the user-facing part of the project, a website where users can log in, upload images for processing, view the results, and manage the system. It is the central hub for all user interactions.

*   **The Database:** This is the system's "memory." It is a robust PostgreSQL database where all the data is securely stored. This includes every license plate detected, user account information, and administrative settings like watchlists.

*   **The AI Models:** A suite of specialized, pre-trained machine learning models are used by the ANPR Engine. Each model has a specific job, such as locating vehicles, identifying license plates, determining vehicle color, and reading the text on the plates.

## 3. How The System Works: A Step-by-Step Flow

The core functionality of the system can be understood through the following step-by-step process, from image upload to final result:

1.  **Image Upload:** The process begins when an administrator uploads an image containing a vehicle through the secure web portal.

2.  **Automated Analysis:** Once uploaded, the image is immediately sent to the ANPR Engine for analysis. The engine performs a series of tasks automatically:
    *   First, it **detects the main vehicle** in the image, identifying its type (e.g., car, bus, truck) and its color. It also attempts to recognize the vehicle's make and model (e.g., 'Toyota Camry').
    *   Next, it specifically **locates the license plate** on the vehicle.
    *   Finally, using advanced Optical Character Recognition (OCR) powered by a large multi-modal AI model, it **reads the alphanumeric characters** on the license plate.

3.  **Data Recording and Storage:** All the information extracted during the analysis is compiled and saved to the database. This includes the license plate number, the confidence of the reading, all the vehicle details (type, color, brand), and the exact time of the detection. The system also saves two images for reference: a cropped image of just the license plate, and the original image with the detected plate highlighted by a bounding box.

4.  **Displaying Results:** The results of the analysis are immediately available on the web application's dashboard. Users can view the annotated image, the cropped plate, and all the associated data in a clear and organized gallery format.

## 4. Key System Features

The project is more than just a plate reader; it is a complete management system with a rich set of features.

*   **Comprehensive Vehicle Identification:** The system goes beyond standard ANPR by capturing not just the plate number, but also the vehicle's type, color, and brand, providing a much richer set of data for each detection.

*   **Secure, Multi-User Portal:** The web application features a full user authentication system. Users can register and log in securely.

*   **Role-Based Access Control:** The system defines two user roles: a standard **User**, who can view detected plates, and an **Admin**, who has full control over the system, including the ability to manage other users and system settings.

*   **Visual Dashboard and Gallery:** The main interface is a dynamic dashboard that displays all detections in a gallery format. This view is paginated and includes filtering capabilities, making it easy to browse and search through historical data.

*   **Powerful Administrative Tools:**
    *   **Watchlist Management:** Administrators can create and manage "watchlists"—special lists of license plates they want to monitor. For example, an admin could create a "Stolen Vehicles" or "Unauthorized Access" list.
    *   **Real-Time Alerts:** This is a critical feature of the system. If a newly detected license plate matches an entry on an active watchlist, the system instantly generates an alert. This alert is then pushed in **real-time** as a notification to the dashboards of all currently logged-in administrators.
    *   **User Management:** Admins have a dedicated panel to view all registered users, manage their roles (e.g., promote a user to an admin), and delete user accounts.
    *   **Usage Analytics:** The system includes a dashboard for admins to view analytics and statistics, such as the number of plates detected over time, breakdowns by vehicle type, and API usage metrics.

## 5. Technologies Used

The project was built using a combination of modern and robust technologies:

*   **Backend & Web:** Python and the Flask framework form the foundation of the web application.
*   **Database:** A PostgreSQL database is used for reliable and structured data storage.
*   **Artificial Intelligence:** The project leverages several AI technologies, including YOLO for high-performance object detection, TensorFlow/Keras for classification tasks, and the Groq API to access large-scale models for advanced OCR and recognition.
*   **Real-time Features:** Flask-SocketIO is used to enable the live, real-time alert notifications for administrators.
*   **Frontend:** The user interface is built with standard web technologies: HTML, CSS, and JavaScript.
