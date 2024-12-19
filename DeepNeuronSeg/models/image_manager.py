import os
from pathlib import Path
from typing import Optional, Tuple, List

class ImageManager:
    """Manages image collection from either a database or filesystem directory"""
    
    SUPPORTED_FORMATS = ('.png', '.tif', '.tiff')
    
    def __init__(self, db=None, dataset_path: Optional[str] = None):
        self.db = db
        self.dataset_path = None
        self.current_index = 0
        self.selected_frame = 0
        
        # Validate inputs
        if not (db or dataset_path):
            raise ValueError("Either db or dataset_path must be provided")
            
        if dataset_path:
            self.set_dataset_path(dataset_path)

    def get_index(self):
        return self.current_index

    def remove_outlier(self):
        self.outlier_files = []

    def set_dataset_path(self, path: str) -> None:
        """
        Set a new dataset directory path
        
        Args:
            path: String path to directory containing images
            
        Raises:
            ValueError: If path doesn't exist or isn't a directory
        """
        if not os.path.exists(path):
            raise ValueError(f"Dataset path does not exist: {path}")
        if not os.path.isdir(path):
            raise ValueError(f"Dataset path is not a directory: {path}")
            
        self.dataset_path = Path(path)
        self.current_index = 0  # Reset index when changing source

    def remove_dataset_path(self) -> None:
        """
        Remove the dataset path and revert to database source
        
        Raises:
            ValueError: If no database was provided and trying to remove the only data source
        """
        if not self.db and self.dataset_path:
            raise ValueError("Cannot remove dataset path when no database is configured")
            
        self.dataset_path = None
        self.current_index = 0  # Reset index when changing source

    def _load_directory_images(self, subdir: Optional[str] = None) -> List[str]:
        """Load image paths from the dataset directory
        
        Args:
            subdir: Optional subdirectory path relative to dataset_path to load images from
            
        Returns:
            List of image file paths
        """
        if not self.dataset_path:
            return []
            
        search_path = self.dataset_path
        if subdir:
            search_path = Path(os.path.join(search_path, subdir))
            if not search_path.exists() or not search_path.is_dir():
                return []
            
        return sorted([
            str(f) for f in search_path.iterdir()
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ])

    def get_item(self, show_masks=False, show_labels=False, no_wrap=False, subdir: Optional[str] = None) -> Tuple[Optional[str], int, int, Optional[List]]:
        """
        Get information about the current image
        
        Returns:
            Tuple containing:
            - Image path (str or None)
            - Current index (int)
            - Total number of items (int)
            - Points data (List or None)
        """
        if self.dataset_path:
            items = self._load_directory_images(subdir)
            points = None  # No points data for directory-based images
        else:
            items = self.db.load_masks() if show_masks else self.db.load_images()
            points = self.db.load_labels()[self.current_index] if show_labels else None
        
        if not items:
            return None, 0, 0, None
            
        if self.current_index >= len(items) and not no_wrap:
            self.current_index = 0
        elif self.current_index >= len(items) and no_wrap:
            return None, 0, 0, None
            
        current_item = items[self.current_index]
        total_items = len(items)
        
        return current_item, self.current_index, total_items, points

    def next_image(self, subdir: Optional[str] = None) -> None:
        """Move to the next image in the sequence"""
        items = self._load_directory_images(subdir) if self.dataset_path else self.db.load_images()
        if items:
            self.current_index = (self.current_index + 1) % len(items)

    def set_index(self, index: int) -> None:
        """Set the current image index"""
        self.current_index = index

    def get_total_images(self, subdir: Optional[str] = None) -> int:
        """Get the total number of images in the current source"""
        if self.dataset_path:
            return len(self._load_directory_images(subdir))
        return len(self.db.load_images() if self.db else [])
    
    def get_images(self, subdir: Optional[str] = None):
        if self.dataset_path:
            images = self._load_directory_images(subdir)
            images = [os.path.splitext(os.path.basename(path))[0] for path in images]
            return images
        images = self.db.load_images() if self.db else []
        images = [os.path.splitext(os.path.basename(path))[0] for path in images]
        return images
