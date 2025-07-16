import unittest
from unittest.mock import patch, MagicMock, call
import os
import tempfile
import numpy as np
import torch
import cv2
import time
import uuid
from threading import Thread


# WICHTIG: Importiere die AKTUELLEN Klassen aus deinen Modulen
# Stelle sicher, dass 'cvbf' in deinem PYTHONPATH ist oder dass du die Tests
# aus dem richtigen Verzeichnis ausführst (z.B. aus 'src/' wenn 'cvbf' darin liegt).
from cvbf.loading.loading import Loading
from cvbf.loading.image_file_loader import ImageFileLoader
from cvbf.loading.image_object_loader import ImageObjectLoader
# from cvbf.loading.lucid_vision_loader import LucidVision # Auskommentiert, da gi nicht verfügbar
from cvbf.loading.video_loader import VideoLoader

# --- Globale Mocks für externe Abhängigkeiten wie gi/Aravis ---
# KOMPLETT AUSKOMMENTIERT, DA gi NICHT VERFÜGBAR IST.
# import sys
# from types import ModuleType

# class MockGi:
#     def require_version(self, name, version):
#         pass

# if "gi" not in sys.modules:
#     sys.modules["gi"] = MockGi()
#     sys.modules["gi.repository"] = ModuleType("gi.repository")
#     sys.modules["gi.repository.Aravis"] = ModuleType("gi.repository.Aravis")

# class MockAravisBuffer:
#     def __init__(self, image_data, frame_id, timestamp, height, width):
#         self._image_data = image_data
#         self._frame_id = frame_id
#         self._timestamp = timestamp
#         self._height = height
#         self._width = width

#     def get_image_data(self):
#         return self._image_data

#     def get_frame_id(self):
#         return self._frame_id

#     def get_timestamp(self):
#         return self._timestamp

#     def get_image_height(self):
#         return self._height

#     def get_image_width(self):
#         return self._width

# class MockAravisStream:
#     def __init__(self):
#         self.pushed_buffers = []
#     def push_buffer(self, buffer):
#         self.pushed_buffers.append(buffer)

# class MockAravisCamera:
#     def __init__(self):
#         self.stream_callback = None
#         self.stream_user_data = None
#         self.stream_instance = MockAravisStream()
#         self.acquisition_started = False

#     def create_stream(self, callback, user_data):
#         self.stream_callback = callback
#         self.stream_user_data = user_data
#         return self.stream_instance

#     def get_payload(self):
#         return 1024 # Dummy payload size

#     def start_acquisition(self):
#         pass

# class UserData:
#     def __init__(self) -> None:
#         self.stream = None
#         self.frame = None
#         self.frame_id = None
#         self.timestamp = None
#         self.locked = False
#         self._device_name = "cpu"
#         self._device_index = 0

#     def set_frame(self, frame):
#         self.frame = frame

#     def get_frame(self):
#         return self.frame

#     def set_timestamp(self, timestamp):
#         self.timestamp = time.time()

#     def get_timestamp(self):
#         return self.timestamp

#     def set_frame_id(self, frame_id):
#         self.frame_id = frame_id

#     def get_frame_id(self):
#         return self.frame_id

#     def get_data(self):
#         if self.locked:
#             return None, None, None
#         else:
#             return self.frame, self.frame_id, self.timestamp
            
#     def get_device(self):
#         return f"{self._device_name}:{self._device_index}"

#     def lock(self):
#         self.locked = True

#     def unlock(self):
#         self.locked = False

# if "gi.repository.Aravis" in sys.modules:
#     sys.modules["gi.repository.Aravis"].Camera.new = MagicMock(return_value=MockAravisCamera())
#     sys.modules["gi.repository.Aravis"].Buffer.new_allocate = MagicMock(return_value=MagicMock(spec=MockAravisBuffer))


# --- Ende der globalen Mocks ---

# Helper-Funktion für den super().__init__ Patch
def init_super_stub(self_instance, *args, **kwargs):
    """
    Ein Stub für die __init__ Methode der Loading-Klasse,
    um die benötigten Attribute in den Mocks zu setzen.
    """
    # Wenn init_super_stub als Patch-Side-Effect auf eine Methode angewendet wird,
    # die *selbst* ein 'self'-Argument hat (wie jede Instanzmethode in den Loader-Tests),
    # dann wird das erste Argument, das init_super_stub erhält, nicht das tatsächliche 'self_instance'
    # des Loaders sein, sondern das `self` des Testfalls (z.B. TestImageFileLoader).
    # Das korrekte self_instance ist dann das *erste Argument, das der gepatchte __init__ erhalten würde*,
    # welches von mock.patch durchgereicht wird.
    # Wenn patch als Klassendekorator verwendet wird, ist das Handling anders.
    # Um es sowohl für methodenbasierte Patches als auch für klassenbasierte Patches zu handhaben,
    # können wir annehmen, dass `self_instance` immer das Objekt ist, dessen `__init__` aufgerufen wurde.
    
    # In der bisherigen Form hat init_super_stub das *erste Argument* des *Original-Aufrufs* bekommen.
    # Da super().__init__(DEVICE_NAME=..., DEVICE_INDEX=...) aufgerufen wird, sind DEVICE_NAME
    # und DEVICE_INDEX **kwargs**.
    # Das Problem war, dass unittest.mock bei einem Patch auf eine Methode das erste Argument
    # (das normalerweise 'self' wäre) als **zusätzliches Positionsargument** an den side_effect übergibt,
    # wenn der side_effect eine Standalone-Funktion ist und nicht eine Methode innerhalb der Testklasse.
    # Das führt dazu, dass 'self_instance' mit dem 'self' des Testfalls gefüllt wird und die tatsächlichen
    # DEVICE_NAME und DEVICE_INDEX in args[0] und args[1] landen (wenn sie positional wären) oder weiterhin in kwargs.

    # Die einfachste und robusteste Lösung ist, das `self_instance` Argument explizit vom Mock zu empfangen,
    # aber gleichzeitig zu erkennen, dass die ursprünglichen Argumente als *args und **kwargs kommen.
    # Da wir es jetzt als Klassendekorator verwenden, ist das `self_instance` das erste Argument des `side_effect`.
    # Die tatsächlichen Argumente von `super().__init__` sind immer `DEVICE_NAME` und `DEVICE_INDEX` als kwargs.

    device_name = kwargs.get("DEVICE_NAME")
    device_index = kwargs.get("DEVICE_INDEX")

    setattr(self_instance, 'id', uuid.uuid1())
    setattr(self_instance, 'DEVICE_NAME', device_name)
    setattr(self_instance, 'DEVICE_INDEX', device_index)
    setattr(self_instance, 'DEVICE', f'{device_name}:{device_index}')
    setattr(self_instance, 'output', [None, None, None])
    setattr(self_instance, 'locked_by', None) # Wichtig für Loading-Tests
    setattr(self_instance, 'stream', None) # Für LucidVision/VideoLoader
    return None

class TestLoading(unittest.TestCase):
    """
    Unit tests for the base Loading class.
    Tests initialization, locking mechanism, and data handling.
    """
    def setUp(self):
        # Für TestLoading brauchen wir keinen Mock auf Loading.__init__,
        # da wir die echte Loading-Klasse testen.
        self.device_name = "cpu"
        self.device_index = 0
        self.loader = Loading(self.device_name, self.device_index)
        self.process_id = self.loader.id # Verwende die eigene ID des Loaders für gültige Operationen
        self.other_process_id = uuid.uuid1() # Eine andere ID für Blockierungsszenarien

    def test_initialization(self):
        """Test if the Loading object is initialized correctly."""
        self.assertIsNotNone(self.loader.id)
        self.assertEqual(self.loader.DEVICE_NAME, self.device_name)
        self.assertEqual(self.loader.DEVICE_INDEX, self.device_index)
        self.assertEqual(self.loader.DEVICE, f'{self.device_name}:{self.device_index}')
        self.assertIsNone(self.loader.stream)
        self.assertEqual(self.loader.output, [None, None, None])
        self.assertIsNone(self.loader.locked_by)

    @patch('builtins.print')
    def test_lock_success(self, mock_print):
        """Test successful locking."""
        self.assertTrue(self.loader.lock(self.process_id))
        self.assertEqual(self.loader.locked_by, self.process_id)
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_lock_failure_already_locked(self, mock_print):
        """Test locking failure when already locked by another process."""
        self.loader.lock(self.other_process_id) # Lock with another process
        self.assertFalse(self.loader.lock(self.process_id)) # Try to lock with self
        self.assertEqual(self.loader.locked_by, self.other_process_id) # Should remain locked by other
        mock_print.assert_called_once_with(f"locking blocked by process: {self.other_process_id}")

    @patch('builtins.print')
    def test_unlock_success(self, mock_print):
        """Test successful unlocking."""
        self.loader.lock(self.process_id)
        self.loader.unlock(self.process_id)
        self.assertIsNone(self.loader.locked_by)
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_unlock_failure_by_other_process(self, mock_print):
        """Test unlocking failure when attempted by a different process."""
        self.loader.lock(self.process_id)
        self.loader.unlock(self.other_process_id) # Try to unlock with other process
        self.assertEqual(self.loader.locked_by, self.process_id) # Should remain locked by self
        mock_print.assert_called_once_with(f"unlocking blocked by process: {self.process_id}")

    @patch('builtins.print')
    def test_set_data_success(self, mock_print):
        """Test successful setting of data when locked by the correct process."""
        self.loader.lock(self.process_id)
        test_frame = "dummy_frame"
        test_frame_id = 123
        test_timestamp = time.time()
        self.loader.set_data(test_frame, test_frame_id, test_timestamp, self.process_id)
        self.assertEqual(self.loader.output, [test_frame, test_frame_id, test_timestamp])
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_set_data_failure_not_locked_by_process(self, mock_print):
        """Test setting data failure when not locked by the specified process."""
        self.loader.lock(self.other_process_id) # Locked by another process
        original_output = list(self.loader.output) # Copy original output
        self.loader.set_data("new_frame", 456, time.time(), self.process_id)
        self.assertEqual(self.loader.output, original_output) # Output should not change
        mock_print.assert_called_once_with(f"setting blocked by process: {self.other_process_id}")

    @patch('builtins.print')
    def test_get_data_success(self, mock_print):
        """Test successful retrieval of data when locked by the correct process."""
        self.loader.lock(self.process_id)
        test_frame = "retrieved_frame"
        test_frame_id = 789
        test_timestamp = time.time()
        self.loader.output = [test_frame, test_frame_id, test_timestamp] # Manually set output
        retrieved_data = self.loader.get_data(self.process_id)
        self.assertEqual(retrieved_data, [test_frame, test_frame_id, test_timestamp])
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_get_data_failure_not_locked_by_process(self, mock_print):
        """Test data retrieval failure when not locked by the specified process."""
        self.loader.lock(self.other_process_id) # Locked by another process
        retrieved_data = self.loader.get_data(self.process_id)
        self.assertEqual(retrieved_data, [None, None, None]) # Should return default
        mock_print.assert_called_once_with(f"getting blocked by process: {self.other_process_id}")

@patch('cvbf.loading.loading.Loading.__init__', side_effect=init_super_stub)
class TestImageFileLoader(unittest.TestCase):
    """
    Unit tests for the ImageFileLoader class.
    Tests image loading and preprocessing.
    """
    @classmethod
    def setUpClass(cls):
        """Create a dummy image file for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dummy_image_path = os.path.join(cls.temp_dir, "dummy_image.png")
        # Create a simple 10x10x3 (BGR) image
        dummy_image_data = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_image_data[:, :, 0] = 255 # Blue channel
        cv2.imwrite(cls.dummy_image_path, dummy_image_data)

    @classmethod
    def tearDownClass(cls):
        """Clean up the dummy image file and directory."""
        os.remove(cls.dummy_image_path)
        os.rmdir(cls.temp_dir)

    # Patch die Methoden, die ImageFileLoader aufruft
    @patch('cv2.imread')
    @patch('torch.from_numpy')
    @patch('cvbf.loading.loading.Loading.lock')
    @patch('cvbf.loading.loading.Loading.unlock')
    @patch('cvbf.loading.loading.Loading.get_data', return_value=[None, None, None])
    def test_init(self, mock_get_data, mock_unlock, mock_lock, mock_from_numpy, mock_imread, mock_super_init): # Reihenfolge ändern, damit mock_super_init am Ende ist
        """Test ImageFileLoader initialization."""
        loader = ImageFileLoader("cpu", 0, self.dummy_image_path) 
        
        mock_super_init.assert_called_once_with(DEVICE_NAME="cpu", DEVICE_INDEX=0)
        self.assertEqual(loader.PATH_TO_IMAGE, self.dummy_image_path)
        
        mock_lock.assert_called_once_with(loader.id)
        mock_get_data.assert_called_once_with(loader.id)
        mock_unlock.assert_called_once_with(loader.id)
        self.assertEqual(loader.output, [None, None, None])

    @patch('cv2.imread')
    @patch('torch.from_numpy')
    @patch('time.time', return_value=12345.6789) # Mock time.time für vorhersagbaren Zeitstempel
    @patch('cvbf.loading.loading.Loading.lock')
    @patch('cvbf.loading.loading.Loading.unlock')
    @patch('cvbf.loading.loading.Loading.set_data')
    def test_load_method(self, mock_set_data, mock_unlock, mock_lock, mock_time, mock_from_numpy, mock_imread, mock_super_init): # Reihenfolge ändern
        """Test the load method's image processing and data setting."""
        loader = ImageFileLoader("cpu", 0, self.dummy_image_path)
        
        # Konfiguriere den Mock für torch.from_numpy und seine gekettelte Aufrufe
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor
        mock_from_numpy.return_value = mock_tensor

        loader.load()

        # ImageFileLoader's __init__ ruft lock/get_data/unlock auf. load() ruft lock/set_data/unlock auf.
        # Daher sollten lock und unlock jeweils zweimal aufgerufen werden.
        expected_lock_calls = [call(loader.id), call(loader.id)]
        mock_lock.assert_has_calls(expected_lock_calls, any_order=False)
        self.assertEqual(mock_lock.call_count, 2)
        
        expected_unlock_calls = [call(loader.id), call(loader.id)]
        mock_unlock.assert_has_calls(expected_unlock_calls, any_order=False)
        self.assertEqual(mock_unlock.call_count, 2)
        
        mock_imread.assert_called_once_with(self.dummy_image_path)
        
        self.assertTrue(mock_from_numpy.called) # Stellt sicher, dass es mindestens einmal aufgerufen wurde
        
        # Überprüfe die Kette von Tensor-Operationen am Rückgabewert von from_numpy
        mock_from_numpy.return_value.to.assert_called_once_with(loader.DEVICE)
        mock_from_numpy.return_value.to.return_value.permute.assert_called_once_with(2, 0, 1)

        mock_set_data.assert_called_once_with(
            mock_tensor, # Der endgültige verarbeitete Mock-Tensor
            0,           # frame_id
            mock_time.return_value, # Zeitstempel
            loader.id    # process_id
        )

@patch('cvbf.loading.loading.Loading.__init__', side_effect=init_super_stub)
class TestImageObjectLoader(unittest.TestCase):
    """
    Unit tests for the ImageObjectLoader class.
    Tests loading of pre-existing image objects (torch.Tensor).
    """
    def test_init(self, mock_super_init): # mock_super_init als letztes Argument
        """Test ImageObjectLoader initialization."""
        loader = ImageObjectLoader("cuda", 1)
        
        mock_super_init.assert_called_once_with(DEVICE_NAME="cuda", DEVICE_INDEX=1)

    @patch('time.time', return_value=98765.4321)
    @patch('cvbf.loading.loading.Loading.lock')
    @patch('cvbf.loading.loading.Loading.unlock')
    @patch('cvbf.loading.loading.Loading.set_data')
    def test_load_method(self, mock_set_data, mock_unlock, mock_lock, mock_time, mock_super_init): # Reihenfolge ändern
        """Test the load method's image object handling and data setting."""
        loader = ImageObjectLoader("cpu", 0)
        
        # Erstelle einen Mock torch.Tensor
        mock_image_tensor = MagicMock(spec=torch.Tensor)
        mock_image_tensor.to.return_value = mock_image_tensor # Simuliere .to() gibt sich selbst zurück

        loader.load(mock_image_tensor)

        mock_lock.assert_called_once_with(loader.id)
        mock_image_tensor.to.assert_called_once_with(loader.DEVICE)
        mock_set_data.assert_called_once_with(
            mock_image_tensor, # Der Mock-Tensor nach dem .to() Aufruf
            0,                 # frame_id
            mock_time.return_value, # Zeitstempel
            loader.id          # process_id
        )
        mock_unlock.assert_called_once_with(loader.id)


# # LucidVision Tests sind auskommentiert, da PyGObject (gi) nicht installiert ist.
# (Unchanged LucidVision tests - not part of the problem)


@patch('cvbf.loading.loading.Loading.__init__', side_effect=init_super_stub)
class TestVideoLoader(unittest.TestCase):
    """
    Unit tests for the VideoLoader class.
    Tests video file loading, frame processing, and threading.
    """
    @classmethod
    def setUpClass(cls):
        """Create a dummy video file for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dummy_video_path = os.path.join(cls.temp_dir, "dummy_video.avi")
        
        # Erstelle ein einfaches 3-Frame-Video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cls.frame_width = 64
        cls.frame_height = 48
        cls.fps = 10
        out = cv2.VideoWriter(cls.dummy_video_path, fourcc, cls.fps, (cls.frame_width, cls.frame_height))
        
        for i in range(3): # 3 Frames
            frame = np.zeros((cls.frame_height, cls.frame_width, 3), dtype=np.uint8)
            frame[:, :, 0] = 255 # Blauer Frame
            if i == 1:
                frame[:, :, 1] = 255 # Grün für den zweiten Frame
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        """Clean up the dummy video file and directory."""
        os.remove(cls.dummy_video_path)
        os.rmdir(cls.temp_dir)

    @patch('cv2.VideoCapture') 
    def test_init_and_load(self, mock_video_capture, mock_super_init): # mock_super_init ans Ende
        """Test VideoLoader initialization and internal __load method."""
        mock_video_capture.return_value = MagicMock() # Stelle sicher, dass VideoCapture ein Mock-Objekt zurückgibt
        
        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)
        
        mock_super_init.assert_called_once_with(DEVICE_NAME="cpu", DEVICE_INDEX=0)
        self.assertEqual(loader.fps, self.fps)
        self.assertEqual(loader.frame_id, 0)
        self.assertFalse(loader.isPlaying)
        
        mock_video_capture.assert_called_once_with(self.dummy_video_path)
        self.assertEqual(loader._VideoLoader__frame_sleep, 1.0 / self.fps)
        self.assertIsNotNone(loader._VideoLoader__videoReader)


    @patch('cv2.VideoCapture')
    @patch('torch.from_numpy')
    @patch('time.time', side_effect=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 3.0]) # Genug Zeitstempel für Init und Frames, inkl. 2D-Frame-Test
    @patch('time.sleep') # Mock time.sleep, um tatsächliche Verzögerungen zu verhindern
    # Patch Loading's lock/unlock/set_data *vor* dem Aufruf von VideoLoader's __init__
    @patch('cvbf.loading.loading.Loading.lock', side_effect=[True, True, True, True, True, True]) # Genug True für Init und 3 Frames, plus den 2D-Frame-Test
    @patch('cvbf.loading.loading.Loading.unlock')
    @patch('cvbf.loading.loading.Loading.set_data')
    def test_play_method_logic(self, mock_set_data, mock_unlock, mock_lock, mock_sleep, mock_time, mock_from_numpy, mock_video_capture, mock_super_init): # Reihenfolge ändern
        """
        Test the core _play method logic without actual threading.
        Simulates reading multiple frames.
        """
        # Setup mock für cv2.VideoCapture.read()
        mock_video_reader = MagicMock()
        mock_video_reader.isOpened.return_value = True
        
        # Simuliere 3 Frames, dann Ende des Videos
        frame1 = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        frame2 = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        frame3 = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Setze side_effect für den ersten Durchlauf von _play
        mock_video_reader.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (True, frame3),
            (False, None) # Ende des Videos
        ]
        mock_video_capture.return_value = mock_video_reader

        # Setup mock für torch.from_numpy und seine gekettelte Aufrufe
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.repeat.return_value = mock_tensor
        mock_from_numpy.return_value = mock_tensor

        # VideoLoader.__init__ wird hier aufgerufen
        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)
        
        # Mocks von __init__-Aufrufen zurücksetzen, bevor _play getestet wird
        mock_lock.reset_mock()
        mock_unlock.reset_mock()
        mock_set_data.reset_mock()
        mock_from_numpy.reset_mock()
        mock_tensor.reset_mock() # Reset mock_tensor calls too

        # Speichere die erwarteten Zeitstempel in einer separaten Liste,
        # bevor mock_time (der Iterator) von loader._play() konsumiert wird.
        expected_time_stamps_for_play_logic = [1.0, 1.1, 1.2] 

        loader._play() # Rufe die private Methode direkt auf zum Unit-Testing

        self.assertFalse(loader.isPlaying)
        self.assertEqual(loader.frame_id, 3) # 3 Frames verarbeitet
        
        expected_set_data_calls = [
            call(mock_tensor, 0, expected_time_stamps_for_play_logic[0], loader.id), 
            call(mock_tensor, 1, expected_time_stamps_for_play_logic[1], loader.id),
            call(mock_tensor, 2, expected_time_stamps_for_play_logic[2], loader.id),
        ]
        mock_set_data.assert_has_calls(expected_set_data_calls)
        self.assertEqual(mock_set_data.call_count, 3)

        # Überprüfe lock/unlock Aufrufe (sollten je 3 sein, da _play 3 Mal läuft)
        self.assertEqual(mock_lock.call_count, 3)
        self.assertEqual(mock_unlock.call_count, 3)

        # Überprüfe time.sleep Aufrufe
        self.assertEqual(mock_sleep.call_count, 3) # Einmal pro Frame
        mock_sleep.assert_has_calls([call(1.0/self.fps)] * 3)

        # Überprüfe torch Operationen
        self.assertEqual(mock_from_numpy.call_count, 3)
        self.assertEqual(mock_tensor.to.call_count, 3)
        self.assertEqual(mock_tensor.permute.call_count, 3)
        
        # Teste den len(image.shape) < 3 Zweig
        mock_set_data.reset_mock()
        mock_unlock.reset_mock()
        mock_lock.reset_mock()
        mock_sleep.reset_mock()
        mock_from_numpy.reset_mock()
        mock_tensor.reset_mock() # Reset mock_tensor calls again

        # Simuliere einen 2D-Frame (z.B. Graustufen)
        frame_2d = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        mock_video_reader.read.side_effect = [
            (True, frame_2d),
            (False, None)
        ]
        mock_from_numpy.return_value = mock_tensor # Mock_from_numpy-Rückgabe zurücksetzen
        loader.frame_id = 0 # frame_id für diesen Untertest zurücksetzen
        
        loader._play()

        mock_tensor.unsqueeze.assert_called_once_with(0)
        mock_tensor.unsqueeze.return_value.repeat.assert_called_once_with(3,1,1)
        mock_set_data.assert_called_once_with(
            mock_tensor.repeat.return_value, # Sollte das Ergebnis von repeat sein
            0,
            3.0, # Der nächste Wert in der time.time() side_effect Liste ist 3.0
            loader.id
        )

    @patch('cvbf.loading.video_loader.Thread') 
    @patch('cv2.VideoCapture')
    def test_play_starts_thread(self, mock_video_capture, mock_thread, mock_super_init): # mock_super_init ans Ende
        """Test that the play method correctly starts a new thread."""
        mock_video_reader_instance = MagicMock()
        mock_video_reader_instance.isOpened.return_value = True
        mock_video_reader_instance.read.return_value = (False, None) # Stoppt die Schleife in _play sofort
        mock_video_capture.return_value = mock_video_reader_instance
        
        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)
        
        # Konfiguriere den Mock-Thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        loader.play()
        
        mock_thread.assert_called_once_with(target=loader._play)
        mock_thread_instance.start.assert_called_once()
        
        # Rufe loader._play() direkt auf, um sicherzustellen, dass die Logik ausgeführt wird
        # und loader.isPlaying korrekt auf False gesetzt wird.
        loader._play()
        self.assertFalse(loader.isPlaying)


    @patch('cvbf.loading.loading.Loading.lock')
    @patch('cvbf.loading.loading.Loading.unlock')
    @patch('cvbf.loading.loading.Loading.get_data', return_value=["frame_data", 1, 1.0])
    def test_next_method(self, mock_get_data, mock_unlock, mock_lock, mock_super_init): # mock_super_init ans Ende
        """Test the next method's data retrieval."""
        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)
        
        loader.next()

        mock_lock.assert_called_once_with(loader.id)
        mock_get_data.assert_called_once_with(loader.id)
        mock_unlock.assert_called_once_with(loader.id)
        self.assertEqual(loader.output, ["frame_data", 1, 1.0])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)