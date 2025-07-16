import uuid

class Loading():
    def __init__(self, DEVICE_NAME, DEVICE_INDEX) -> None:
        """
        Initialize a new Loading object with the given device name and index.

        Parameters:
        DEVICE_NAME (str): The name of the device.
        DEVICE_INDEX (int): The index of the device.

        Returns:
        None
        """
        self.id = uuid.uuid1()
        self.DEVICE_NAME = DEVICE_NAME
        self.DEVICE_INDEX = DEVICE_INDEX
        self.DEVICE = f'{self.DEVICE_NAME}:{self.DEVICE_INDEX}'
        self.stream = None
        self.output = [None, None, None]
        self.locked_by = None


    def lock(self, process) -> bool:
        """
        Lock the loading object for exclusive access by a specific process.

        Parameters:
        process (str): The identifier of the process attempting to lock the object.

        Returns:
        bool: True if the object is successfully locked, False otherwise. If the object is already locked by another process,
        this method will print a message indicating the blocking process and return False.
        """
        if self.locked_by == None:
            self.locked_by = process
            return True
        else:
            print("locking blocked by process: "+str(self.locked_by))
            return False


    def unlock(self, process):
        """
        Release the lock on the loading object, allowing other processes to access it.

        Parameters:
        process (str): The identifier of the process attempting to unlock the object.

        Returns:
        None

        If the object is currently locked by a different process, a message indicating the blocking process is printed,
        and the method does not release the lock.
        """
        if self.locked_by == process:
            self.locked_by = None
        else:
            print("unlocking blocked by process: "+str(self.locked_by))


    def set_data(self, frame, frame_id, timestamp, process):
        """
        Set the frame, frame ID, and timestamp associated with the loading object.

        This method checks if the loading object is currently locked by the specified process.
        If the object is locked, the frame, frame ID, and timestamp are updated.
        If the object is not locked by the specified process, a message is printed indicating the blocking process.

        Parameters:
        frame (object): The frame data to be associated with the loading object.
        frame_id (int): The unique identifier for the frame.
        timestamp (float): The timestamp associated with the frame.
        process (str): The identifier of the process attempting to set the data.

        Returns:
        None
        """
        if self.locked_by == process:
            self.output = [frame, frame_id, timestamp]
        else:
            print("setting blocked by process: "+str(self.locked_by))


    def get_data(self, process):
        """
        Retrieve the frame, frame ID, and timestamp associated with the loading object.

        Parameters:
        process (str): The identifier of the process attempting to retrieve the data.

        Returns:
        tuple: A tuple containing the frame, frame ID, and timestamp. If the object is not locked by the specified process,
        this method will print a message indicating the blocking process and return None for all values.
        """
        if self.locked_by == process:
            return self.output
        else:
            print("getting blocked by process: "+str(self.locked_by))
            return [None, None, None]

