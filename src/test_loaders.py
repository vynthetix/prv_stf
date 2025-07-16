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

from cvbf.loading.loading import Loading
from cvbf.loading.image_file_loader import ImageFileLoader
from cvbf.loading.image_object_loader import ImageObjectLoader
from cvbf.loading.video_loader import VideoLoader


def init_super_stub(self_instance, *args, **kwargs):
    """
    Ein Stub für die __init__ Methode der Loading-Klasse, um die
    benötigten Attribute in den Mocks zu setzen.
    """
    device_name = kwargs.get("DEVICE_NAME")
    device_index = kwargs.get("DEVICE_INDEX")
    setattr(self_instance, "id", uuid.uuid1())
    setattr(self_instance, "DEVICE_NAME", device_name)
    setattr(self_instance, "DEVICE_INDEX", device_index)
    setattr(self_instance, "DEVICE", f"{device_name}:{device_index}")
    setattr(self_instance, "output", [None, None, None])
    setattr(self_instance, "locked_by", None)
    setattr(self_instance, "stream", None)
    return None


class TestLoading(unittest.TestCase):
    """
    Unit tests for the base Loading class. Tests initialization, locking
    mechanism, and data handling.
    """

    def setUp(self):
        self.device_name = "cpu"
        self.device_index = 0
        self.loader = Loading(self.device_name, self.device_index)
        self.process_id = self.loader.id
        self.other_process_id = uuid.uuid1()

    def test_initialization(self):
        """Test if the Loading object is initialized correctly."""
        self.assertIsNotNone(self.loader.id)
        self.assertEqual(self.loader.DEVICE_NAME, self.device_name)
        self.assertEqual(self.loader.DEVICE_INDEX, self.device_index)
        self.assertEqual(
            self.loader.DEVICE, f"{self.device_name}:{self.device_index}"
        )
        self.assertIsNone(self.loader.stream)
        self.assertEqual(self.loader.output, [None, None, None])
        self.assertIsNone(self.loader.locked_by)

    @patch("builtins.print")
    def test_lock_success(self, mock_print):
        """Test successful locking."""
        self.assertTrue(self.loader.lock(self.process_id))
        self.assertEqual(self.loader.locked_by, self.process_id)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_lock_failure_already_locked(self, mock_print):
        """Test locking failure when already locked by another process."""
        self.loader.lock(self.other_process_id)
        self.assertFalse(self.loader.lock(self.process_id))
        self.assertEqual(self.loader.locked_by, self.other_process_id)
        mock_print.assert_called_once_with(
            f"locking blocked by process: {self.other_process_id}"
        )

    @patch("builtins.print")
    def test_unlock_success(self, mock_print):
        """Test successful unlocking."""
        self.loader.lock(self.process_id)
        self.loader.unlock(self.process_id)
        self.assertIsNone(self.loader.locked_by)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_unlock_failure_by_other_process(self, mock_print):
        """Test unlocking failure when attempted by a different process."""
        self.loader.lock(self.process_id)
        self.loader.unlock(self.other_process_id)
        self.assertEqual(self.loader.locked_by, self.process_id)
        mock_print.assert_called_once_with(
            f"unlocking blocked by process: {self.process_id}"
        )

    @patch("builtins.print")
    def test_set_data_success(self, mock_print):
        """
        Test successful setting of data when locked by the correct process.
        """
        self.loader.lock(self.process_id)
        test_frame = "dummy_frame"
        test_frame_id = 123
        test_timestamp = time.time()
        self.loader.set_data(
            test_frame, test_frame_id, test_timestamp, self.process_id
        )
        self.assertEqual(
            self.loader.output, [test_frame, test_frame_id, test_timestamp]
        )
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_set_data_failure_not_locked_by_process(self, mock_print):
        """
        Test setting data failure when not locked by the specified process.
        """
        self.loader.lock(self.other_process_id)
        original_output = list(self.loader.output)
        self.loader.set_data("new_frame", 456, time.time(), self.process_id)
        self.assertEqual(self.loader.output, original_output)
        mock_print.assert_called_once_with(
            f"setting blocked by process: {self.other_process_id}"
        )

    @patch("builtins.print")
    def test_get_data_success(self, mock_print):
        """
        Test successful retrieval of data when locked by the correct process.
        """
        self.loader.lock(self.process_id)
        test_frame = "retrieved_frame"
        test_frame_id = 789
        test_timestamp = time.time()
        self.loader.output = [test_frame, test_frame_id, test_timestamp]
        retrieved_data = self.loader.get_data(self.process_id)
        self.assertEqual(
            retrieved_data, [test_frame, test_frame_id, test_timestamp]
        )
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_get_data_failure_not_locked_by_process(self, mock_print):
        """
        Test data retrieval failure when not locked by the specified process.
        """
        self.loader.lock(self.other_process_id)
        retrieved_data = self.loader.get_data(self.process_id)
        self.assertEqual(retrieved_data, [None, None, None])
        mock_print.assert_called_once_with(
            f"getting blocked by process: {self.other_process_id}"
        )


@patch(
    "cvbf.loading.loading.Loading.__init__", side_effect=init_super_stub
)
class TestImageFileLoader(unittest.TestCase):
    """
    Unit tests for the ImageFileLoader class. Tests image loading and
    preprocessing.
    """

    @classmethod
    def setUpClass(cls):
        """Create a dummy image file for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dummy_image_path = os.path.join(cls.temp_dir, "dummy_image.png")
        dummy_image_data = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_image_data[:, :, 0] = 255
        cv2.imwrite(cls.dummy_image_path, dummy_image_data)

    @classmethod
    def tearDownClass(cls):
        """Clean up the dummy image file and directory."""
        os.remove(cls.dummy_image_path)
        os.rmdir(cls.temp_dir)

    @patch("cv2.imread")
    @patch("torch.from_numpy")
    @patch("cvbf.loading.loading.Loading.lock")
    @patch("cvbf.loading.loading.Loading.unlock")
    @patch(
        "cvbf.loading.loading.Loading.get_data", return_value=[None, None, None]
    )
    def test_init(
        self,
        mock_get_data,
        mock_unlock,
        mock_lock,
        mock_from_numpy,
        mock_imread,
        mock_super_init,
    ):
        """Test ImageFileLoader initialization."""
        loader = ImageFileLoader("cpu", 0, self.dummy_image_path)
        mock_super_init.assert_called_once_with(
            DEVICE_NAME="cpu", DEVICE_INDEX=0
        )
        self.assertEqual(loader.PATH_TO_IMAGE, self.dummy_image_path)
        mock_lock.assert_called_once_with(loader.id)
        mock_get_data.assert_called_once_with(loader.id)
        mock_unlock.assert_called_once_with(loader.id)
        self.assertEqual(loader.output, [None, None, None])

    @patch("cv2.imread")
    @patch("torch.from_numpy")
    @patch("time.time", return_value=12345.6789)
    @patch("cvbf.loading.loading.Loading.lock")
    @patch("cvbf.loading.loading.Loading.unlock")
    @patch("cvbf.loading.loading.Loading.set_data")
    def test_load_method(
        self,
        mock_set_data,
        mock_unlock,
        mock_lock,
        mock_time,
        mock_from_numpy,
        mock_imread,
        mock_super_init,
    ):
        """Test the load method's image processing and data setting."""
        loader = ImageFileLoader("cpu", 0, self.dummy_image_path)
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor
        mock_from_numpy.return_value = mock_tensor
        loader.load()

        expected_lock_calls = [call(loader.id), call(loader.id)]
        mock_lock.assert_has_calls(expected_lock_calls, any_order=False)
        self.assertEqual(mock_lock.call_count, 2)
        expected_unlock_calls = [call(loader.id), call(loader.id)]
        mock_unlock.assert_has_calls(expected_unlock_calls, any_order=False)
        self.assertEqual(mock_unlock.call_count, 2)

        mock_imread.assert_called_once_with(self.dummy_image_path)
        self.assertTrue(mock_from_numpy.called)

        mock_from_numpy.return_value.to.assert_called_once_with(loader.DEVICE)
        mock_from_numpy.return_value.to.return_value.permute.assert_called_once_with(
            2, 0, 1
        )
        mock_set_data.assert_called_once_with(
            mock_tensor,
            0,
            mock_time.return_value,
            loader.id,
        )


@patch(
    "cvbf.loading.loading.Loading.__init__", side_effect=init_super_stub
)
class TestImageObjectLoader(unittest.TestCase):
    """
    Unit tests for the ImageObjectLoader class. Tests loading of pre existing
    image objects (torch.Tensor).
    """

    def test_init(self, mock_super_init):
        """Test ImageObjectLoader initialization."""
        loader = ImageObjectLoader("cuda", 1)
        mock_super_init.assert_called_once_with(
            DEVICE_NAME="cuda", DEVICE_INDEX=1
        )

    @patch("time.time", return_value=98765.4321)
    @patch("cvbf.loading.loading.Loading.lock")
    @patch("cvbf.loading.loading.Loading.unlock")
    @patch("cvbf.loading.loading.Loading.set_data")
    def test_load_method(
        self,
        mock_set_data,
        mock_unlock,
        mock_lock,
        mock_time,
        mock_super_init,
    ):
        """
        Test the load method's image object handling and data setting.
        """
        loader = ImageObjectLoader("cpu", 0)
        mock_image_tensor = MagicMock(spec=torch.Tensor)
        mock_image_tensor.to.return_value = mock_image_tensor
        loader.load(mock_image_tensor)

        mock_lock.assert_called_once_with(loader.id)
        mock_image_tensor.to.assert_called_once_with(loader.DEVICE)
        mock_set_data.assert_called_once_with(
            mock_image_tensor,
            0,
            mock_time.return_value,
            loader.id,
        )
        mock_unlock.assert_called_once_with(loader.id)


@patch(
    "cvbf.loading.loading.Loading.__init__", side_effect=init_super_stub
)
class TestVideoLoader(unittest.TestCase):
    """
    Unit tests for the VideoLoader class. Tests video file loading, frame
    processing, and threading.
    """

    @classmethod
    def setUpClass(cls):
        """Create a dummy video file for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dummy_video_path = os.path.join(cls.temp_dir, "dummy_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cls.frame_width = 64
        cls.frame_height = 48
        cls.fps = 10
        out = cv2.VideoWriter(
            cls.dummy_video_path, fourcc, cls.fps, (cls.frame_width, cls.frame_height)
        )
        for i in range(3):
            frame = np.zeros(
                (cls.frame_height, cls.frame_width, 3), dtype=np.uint8
            )
            frame[:, :, 0] = 255
            if i == 1:
                frame[:, :, 1] = 255
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        """Clean up the dummy video file and directory."""
        os.remove(cls.dummy_video_path)
        os.rmdir(cls.temp_dir)

    @patch("cv2.VideoCapture")
    def test_init_and_load(self, mock_video_capture, mock_super_init):
        """Test VideoLoader initialization and internal __load method."""
        mock_video_capture.return_value = MagicMock()
        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)
        mock_super_init.assert_called_once_with(
            DEVICE_NAME="cpu", DEVICE_INDEX=0
        )
        self.assertEqual(loader.fps, self.fps)
        self.assertEqual(loader.frame_id, 0)
        self.assertFalse(loader.isPlaying)
        mock_video_capture.assert_called_once_with(self.dummy_video_path)
        self.assertEqual(loader._VideoLoader__frame_sleep, 1.0 / self.fps)
        self.assertIsNotNone(loader._VideoLoader__videoReader)

    @patch("cv2.VideoCapture")
    @patch("torch.from_numpy")
    @patch("time.time", side_effect=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 3.0])
    @patch("time.sleep")
    @patch(
        "cvbf.loading.loading.Loading.lock",
        side_effect=[True, True, True, True, True, True],
    )
    @patch("cvbf.loading.loading.Loading.unlock")
    @patch("cvbf.loading.loading.Loading.set_data")
    def test_play_method_logic(
        self,
        mock_set_data,
        mock_unlock,
        mock_lock,
        mock_sleep,
        mock_time,
        mock_from_numpy,
        mock_video_capture,
        mock_super_init,
    ):
        """
        Test the core _play method logic without actual threading.
        Simulates reading multiple frames.
        """
        mock_video_reader = MagicMock()
        mock_video_reader.isOpened.return_value = True
        frame1 = np.zeros(
            (self.frame_height, self.frame_width, 3), dtype=np.uint8
        )
        frame2 = np.zeros(
            (self.frame_height, self.frame_width, 3), dtype=np.uint8
        )
        frame3 = np.zeros(
            (self.frame_height, self.frame_width, 3), dtype=np.uint8
        )
        mock_video_reader.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (True, frame3),
            (False, None),
        ]
        mock_video_capture.return_value = mock_video_reader

        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.repeat.return_value = mock_tensor
        mock_from_numpy.return_value = mock_tensor

        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)

        mock_lock.reset_mock()
        mock_unlock.reset_mock()
        mock_set_data.reset_mock()
        mock_from_numpy.reset_mock()
        mock_tensor.reset_mock()

        expected_time_stamps_for_play_logic = [1.0, 1.1, 1.2]
        loader._play()

        self.assertFalse(loader.isPlaying)
        self.assertEqual(loader.frame_id, 3)
        expected_set_data_calls = [
            call(mock_tensor, 0, expected_time_stamps_for_play_logic[0], loader.id),
            call(mock_tensor, 1, expected_time_stamps_for_play_logic[1], loader.id),
            call(mock_tensor, 2, expected_time_stamps_for_play_logic[2], loader.id),
        ]
        mock_set_data.assert_has_calls(expected_set_data_calls)
        self.assertEqual(mock_set_data.call_count, 3)

        self.assertEqual(mock_lock.call_count, 3)
        self.assertEqual(mock_unlock.call_count, 3)

        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_has_calls([call(1.0 / self.fps)] * 3)

        self.assertEqual(mock_from_numpy.call_count, 3)
        self.assertEqual(mock_tensor.to.call_count, 3)
        self.assertEqual(mock_tensor.permute.call_count, 3)

        mock_set_data.reset_mock()
        mock_unlock.reset_mock()
        mock_lock.reset_mock()
        mock_sleep.reset_mock()
        mock_from_numpy.reset_mock()
        mock_tensor.reset_mock()

        frame_2d = np.zeros(
            (self.frame_height, self.frame_width), dtype=np.uint8
        )
        mock_video_reader.read.side_effect = [
            (True, frame_2d),
            (False, None),
        ]
        mock_from_numpy.return_value = mock_tensor
        loader.frame_id = 0
        loader._play()
        mock_tensor.unsqueeze.assert_called_once_with(0)
        mock_tensor.unsqueeze.return_value.repeat.assert_called_once_with(
            3, 1, 1
        )
        mock_set_data.assert_called_once_with(
            mock_tensor.repeat.return_value,
            0,
            3.0,
            loader.id,
        )

    @patch("cvbf.loading.video_loader.Thread")
    @patch("cv2.VideoCapture")
    def test_play_starts_thread(
        self, mock_video_capture, mock_thread, mock_super_init
    ):
        """Test that the play method correctly starts a new thread."""
        mock_video_reader_instance = MagicMock()
        mock_video_reader_instance.isOpened.return_value = True
        mock_video_reader_instance.read.return_value = (False, None)
        mock_video_capture.return_value = mock_video_reader_instance

        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        loader.play()
        mock_thread.assert_called_once_with(target=loader._play)
        mock_thread_instance.start.assert_called_once()

        loader._play()
        self.assertFalse(loader.isPlaying)

    @patch("cvbf.loading.loading.Loading.lock")
    @patch("cvbf.loading.loading.Loading.unlock")
    @patch(
        "cvbf.loading.loading.Loading.get_data",
        return_value=["frame_data", 1, 1.0],
    )
    def test_next_method(
        self, mock_get_data, mock_unlock, mock_lock, mock_super_init
    ):
        """Test the next method's data retrieval."""
        loader = VideoLoader("cpu", 0, self.dummy_video_path, self.fps)
        loader.next()
        mock_lock.assert_called_once_with(loader.id)
        mock_get_data.assert_called_once_with(loader.id)
        mock_unlock.assert_called_once_with(loader.id)
        self.assertEqual(loader.output, ["frame_data", 1, 1.0])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
