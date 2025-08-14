# Requirements for Our Project

## I. Core Computing Hardware

1. **Raspberry Pi 4 Model B:**We will use a Raspberry Pi 4 Model B, ideally with 4GB or 8GB of RAM, to handle the AI models smoothly and efficiently.
2. **Heatsink Kit + Fan:**Since the Pi 4 tends to heat up when running computer vision models continuously, a good heatsink kit and fan are essential to keep the system cool and prevent throttling.
3. **Case for Raspberry Pi 4:**
   To protect our board and ensure durability, we’ll need a compatible case that accommodates the cooling solution.

## II. Camera System

4. **Camera Options:**
   - **Preferred Option:** We plan to use the Raspberry Pi Camera Module 3, which offers excellent quality, autofocus, and a direct CSI connection.
   - **Alternative:** We could also consider a high-quality USB webcam (with 1080p resolution and reliable low-light performance) if needed, ensuring it works well with Linux and OpenCV.

## III. Storage

5. **MicroSD Card:**
   We’ll require a high-quality, high-speed MicroSD card with at least 32GB of capacity (64GB is recommended) to store the operating system, libraries, code, output videos, and logs. It should be at least Class 10/U1/A1, though A2/U3 is strongly recommended for better performance.

## IV. Power

6. **Power Supply:**
   - **Preferred:** We’ll use the Official Raspberry Pi 4 USB-C Power Supply to guarantee a stable voltage and current.
   - **Alternative:** A reliable third-party USB-C Power Delivery (PD) power supply rated at least 5V/3A can also work well.

## V. Networking

7. **Network Connection:**
   We need reliable network access, either via the built-in WiFi on the Pi 4 or through an Ethernet cable connected to a router or network switch. This connection is crucial for installing packages, connecting to the database, SSH access, and potentially hosting a web interface.

## VI. Database

8. **PostgreSQL Database Access:**Our project requires a running PostgreSQL database instance. We can choose from the following options:
   - **Cloud Service (Recommended):** We might use a free tier from providers such as ElephantSQL, Neon, or Supabase.
   - **On Another Computer:** Alternatively, we could install the PostgreSQL server on our Windows 11 PC or another machine on the same network as the Pi.
   - **On the Pi (Optional):** While possible, running PostgreSQL directly on the Pi 4 could affect performance when combined with other heavy processes.

## VII. Peripherals (For Setup, Development, and Direct Display)

9. **Monitor:**Any monitor with an HDMI input will work well for our setup.
10. **Micro-HDMI to HDMI Cable/Adapter:**Necessary for connecting the Pi 4 to a monitor.
11. **USB Keyboard and Mouse:**These will help us during the initial setup and debugging process.
12. **Our Windows 11 PC:**We will use our PC for:

    - Flashing the SD card with Raspberry Pi OS using Raspberry Pi Imager.
    - Connecting to the Pi via SSH (which is highly recommended).
    - Accessing the database through client tools.
    - Viewing the web interface if we decide to build one.
    - Transferring code and video files as needed.

## VIII. Software (Installed on the Pi)

13. **Operating System:**We will install Raspberry Pi OS (64-bit is recommended) on the MicroSD card.
14. **Project Code:**Our project code, such as `anpr_new.py` and any supporting files/models, will be deployed on the Pi.
15. **Required Libraries:**We’ll need Python 3 along with libraries like Pip, Virtualenv, OpenCV, Ultralytics (YOLO), TensorFlow/Keras, PaddleOCR, Psycopg2-binary, python-dotenv, and NumPy. These can be installed via `apt` and `pip`.
16. **(Optional) Web Framework:**If we build a web interface later, we might use Flask or FastAPI.
17. **(Optional) Database Server:**
    Should we choose to run PostgreSQL directly on the Pi, we would install it on the Raspberry Pi OS, though this may be more suitable for experimental setups rather than final deployment.

---

## After Getting Raspberry Pi OS on the Pi

1. **Clone Our Project Code:**We will clone our project repository onto the Pi.
2. **Create a Python Virtual Environment:**In the project directory, we’ll set up a virtual environment to manage our dependencies.
3. **Install Necessary Dependencies:**
   We will install all required system libraries using `apt` and Python packages using `pip` within the virtual environment.

---

### Additional Points

**PostgreSQL on the Pi:**
We can install and run PostgreSQL directly on the Raspberry Pi 4 using `sudo apt install postgresql`. However, because running PostgreSQL alongside our computationally heavy models might impact performance, we might consider using a cloud database service or hosting PostgreSQL on our laptop. Offloading the database will free up resources on the Pi for our core vision processing tasks. For a final year project where a reliable demonstration is key, using an external database solution is often a safer choice, although running it locally is still an option for testing.

**Camera Recommendation:**For our ANPR project, choosing the right camera is critical. We need to capture clear images of license plates under various conditions:

- The **Raspberry Pi Camera Module 3** is recommended due to its direct integration with the Pi’s CSI port, good image quality, and autofocus capability.
- Alternatively, a quality USB webcam (for example, a model like the Logitech C920) could be used if we need more flexibility in positioning or additional features.

**Network Connection:**
A reliable network connection is essential for installing dependencies, connecting to our database (whether it’s cloud-based or hosted on another machine), and for remote access via SSH. Additionally, if we build a web interface later, the network will allow us to serve web pages to other devices.

**Displaying Detections for Videos:**Our script will:

- Display live detections in real time on a monitor connected to the Pi (great for debugging).
- Save processed video frames with annotations to an output file for later review.
- Save structured detection data (including license plate information) to our PostgreSQL database, which can be viewed using database tools like pgAdmin or DBeaver.
- Optionally, we might develop a web interface to show these detections and statistics directly from the Pi.

---

### Updated Components List

**Essential Core Components:**

- Raspberry Pi 4 Model B (preferably 4GB or 8GB RAM)
- High-quality MicroSD Card (32GB minimum, 64GB recommended)
- Official Raspberry Pi 4 USB-C Power Supply (or a reliable 5V/3A+ USB-C alternative)
- Suitable Camera (preferably Raspberry Pi Camera Module 3 or a good quality USB webcam)
- Heatsink Kit and Fan for the Pi 4
- Protective Case for the Pi
- Reliable network access (WiFi or Ethernet)

**Database Requirements:**

- Access to a PostgreSQL database (using a cloud service like ElephantSQL/Neon or running it on another computer on our network)

**For Setup and Development:**

- Monitor with HDMI input, micro-HDMI to HDMI cable/adapter, USB keyboard and mouse
- Our Windows 11 PC for flashing the SD card, SSH access, database management, and code transfers

**Software Requirements on the Pi:**

- Raspberry Pi OS (64-bit recommended)
- Our project code (e.g., `anpr_new.py` and supporting files/models)
- Required Python libraries (OpenCV, YOLO, PaddleOCR, etc.)
- (Optional) Web framework like Flask or FastAPI for a web interface
- (Optional) PostgreSQL server if we choose to host the database locally
