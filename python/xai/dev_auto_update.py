"""
DIREWOLF Development Auto-Update System

Automatically rebuilds and updates the running DIREWOLF application
when source code changes are detected during development.
"""

import os
import sys
import time
import subprocess
import threading
import importlib
from typing import Dict, List, Callable, Optional, Set
from pathlib import Path
from datetime import datetime
import hashlib
import json


class FileWatcher:
    """
    Watches source files for changes and triggers rebuilds.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.watch_paths = config.get('watch_paths', [])
        self.file_extensions = config.get('file_extensions', ['.cpp', '.hpp', '.py', '.qml'])
        self.ignore_patterns = config.get('ignore_patterns', ['__pycache__', '.git', 'build'])
        
        # File tracking
        self.file_hashes: Dict[str, str] = {}
        self.last_scan_time = 0
        
        # Callbacks
        self.on_change_callback: Optional[Callable] = None
        
        # Control
        self.running = False
        self.scan_thread: Optional[threading.Thread] = None
        
        print("[DevUpdate] File watcher initialized")
    
    def start_watching(self, callback: Callable[[List[str]], None]):
        """
        Start watching for file changes.
        
        Args:
            callback: Function to call when changes detected
        """
        self.on_change_callback = callback
        self.running = True
        
        # Initial scan
        self._scan_files()
        
        # Start monitoring thread
        self.scan_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.scan_thread.start()
        
        print("[DevUpdate] Started watching for file changes")
    
    def stop_watching(self):
        """Stop watching for changes."""
        self.running = False
        if self.scan_thread:
            self.scan_thread.join(timeout=1.0)
        
        print("[DevUpdate] Stopped watching for file changes")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                changed_files = self._scan_files()
                
                if changed_files and self.on_change_callback:
                    self.on_change_callback(changed_files)
                
                time.sleep(self.config.get('scan_interval', 2.0))
                
            except Exception as e:
                print(f"[DevUpdate] Error in monitor loop: {e}")
                time.sleep(5.0)
    
    def _scan_files(self) -> List[str]:
        """Scan files for changes."""
        changed_files = []
        current_time = time.time()
        
        for watch_path in self.watch_paths:
            path = Path(watch_path)
            if not path.exists():
                continue
            
            for file_path in self._get_files_recursive(path):
                if self._should_ignore_file(file_path):
                    continue
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(file_path)
                file_key = str(file_path)
                
                # Check if file changed
                if file_key in self.file_hashes:
                    if self.file_hashes[file_key] != file_hash:
                        changed_files.append(file_key)
                        self.file_hashes[file_key] = file_hash
                        print(f"[DevUpdate] Detected change: {file_path.name}")
                else:
                    # New file
                    self.file_hashes[file_key] = file_hash
        
        self.last_scan_time = current_time
        return changed_files
    
    def _get_files_recursive(self, path: Path) -> List[Path]:
        """Get all files recursively."""
        files = []
        
        try:
            if path.is_file():
                if path.suffix in self.file_extensions:
                    files.append(path)
            elif path.is_dir():
                for item in path.iterdir():
                    if not self._should_ignore_path(item):
                        files.extend(self._get_files_recursive(item))
        except PermissionError:
            pass  # Skip inaccessible directories
        
        return files
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        file_str = str(file_path)
        
        for pattern in self.ignore_patterns:
            if pattern in file_str:
                return True
        
        return file_path.suffix not in self.file_extensions
    
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)
        
        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""


class BuildSystem:
    """
    Handles building C++ and Python components.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.project_root = Path(config.get('project_root', '.'))
        self.build_dir = self.project_root / config.get('build_dir', 'build')
        self.cmake_args = config.get('cmake_args', [])
        
        print("[DevUpdate] Build system initialized")
    
    def build_cpp_components(self, changed_files: List[str]) -> bool:
        """
        Build C++ components.
        
        Args:
            changed_files: List of changed files
            
        Returns:
            True if build successful
        """
        print("[DevUpdate] Building C++ components...")
        
        try:
            # Ensure build directory exists
            self.build_dir.mkdir(exist_ok=True)
            
            # Run CMake configure (only if needed)
            cmake_cache = self.build_dir / "CMakeCache.txt"
            if not cmake_cache.exists():
                print("[DevUpdate] Running CMake configure...")
                
                cmake_cmd = [
                    'cmake',
                    str(self.project_root),
                    '-DCMAKE_BUILD_TYPE=Debug',
                    '-DBUILD_TESTING=ON'
                ] + self.cmake_args
                
                result = subprocess.run(
                    cmake_cmd,
                    cwd=self.build_dir,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"[DevUpdate] CMake configure failed: {result.stderr}")
                    return False
            
            # Run build
            print("[DevUpdate] Running build...")
            
            build_cmd = ['cmake', '--build', '.', '--parallel']
            
            result = subprocess.run(
                build_cmd,
                cwd=self.build_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[DevUpdate] Build failed: {result.stderr}")
                return False
            
            print("[DevUpdate] C++ build successful")
            return True
            
        except Exception as e:
            print(f"[DevUpdate] Build error: {e}")
            return False
    
    def get_changed_file_types(self, changed_files: List[str]) -> Set[str]:
        """
        Determine which types of files changed.
        
        Args:
            changed_files: List of changed file paths
            
        Returns:
            Set of file types ('cpp', 'python', 'qml', etc.)
        """
        file_types = set()
        
        for file_path in changed_files:
            path = Path(file_path)
            
            if path.suffix in ['.cpp', '.hpp', '.h', '.cc']:
                file_types.add('cpp')
            elif path.suffix == '.py':
                file_types.add('python')
            elif path.suffix == '.qml':
                file_types.add('qml')
        
        return file_types


class HotReloader:
    """
    Hot reloads Python modules without restarting.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.loaded_modules: Dict[str, float] = {}
        
        print("[DevUpdate] Hot reloader initialized")
    
    def reload_python_modules(self, changed_files: List[str]):
        """
        Reload changed Python modules.
        
        Args:
            changed_files: List of changed Python files
        """
        for file_path in changed_files:
            path = Path(file_path)
            
            if path.suffix != '.py':
                continue
            
            # Convert file path to module name
            module_name = self._file_to_module(path)
            
            if not module_name:
                continue
            
            try:
                # Check if module is already loaded
                if module_name in sys.modules:
                    print(f"[DevUpdate] Reloading module: {module_name}")
                    importlib.reload(sys.modules[module_name])
                else:
                    print(f"[DevUpdate] Loading new module: {module_name}")
                    importlib.import_module(module_name)
                
                self.loaded_modules[module_name] = time.time()
                
            except Exception as e:
                print(f"[DevUpdate] Failed to reload {module_name}: {e}")
    
    def _file_to_module(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to Python module name.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Module name or None
        """
        try:
            # Get relative path from project root
            project_root = Path(self.config.get('project_root', '.'))
            rel_path = file_path.relative_to(project_root)
            
            # Convert to module name
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            module_name = '.'.join(parts)
            
            return module_name
            
        except Exception:
            return None


class DevAutoUpdate:
    """
    Main development auto-update system.
    
    Coordinates file watching, building, and hot reloading.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize auto-update system.
        
        Args:
            config: Configuration dict with:
                - project_root: Root directory of project
                - watch_paths: Paths to watch for changes
                - build_dir: Build directory for C++
                - enabled: Whether auto-update is enabled
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Components
        self.file_watcher = FileWatcher(config)
        self.build_system = BuildSystem(config)
        self.hot_reloader = HotReloader(config)
        
        # State
        self.is_building = False
        self.last_build_time = 0
        self.build_cooldown = config.get('build_cooldown', 5.0)  # Seconds
        
        # Callbacks
        self.on_update_complete: Optional[Callable] = None
        
        print("[DevUpdate] DIREWOLF Development Auto-Update System initialized")
    
    def start(self, on_update_complete: Optional[Callable] = None):
        """
        Start auto-update system.
        
        Args:
            on_update_complete: Callback when update completes
        """
        if not self.enabled:
            print("[DevUpdate] Auto-update disabled")
            return
        
        self.on_update_complete = on_update_complete
        
        # Start file watcher
        self.file_watcher.start_watching(self._on_files_changed)
        
        print("[DevUpdate] Auto-update system started")
    
    def stop(self):
        """Stop auto-update system."""
        self.file_watcher.stop_watching()
        print("[DevUpdate] Auto-update system stopped")
    
    def _on_files_changed(self, changed_files: List[str]):
        """
        Handle file changes.
        
        Args:
            changed_files: List of changed file paths
        """
        # Check build cooldown
        current_time = time.time()
        if current_time - self.last_build_time < self.build_cooldown:
            print("[DevUpdate] Build cooldown active, skipping...")
            return
        
        if self.is_building:
            print("[DevUpdate] Build already in progress, skipping...")
            return
        
        print(f"[DevUpdate] Processing {len(changed_files)} changed files")
        
        # Determine what changed
        file_types = self.build_system.get_changed_file_types(changed_files)
        
        self.is_building = True
        self.last_build_time = current_time
        
        try:
            # Handle C++ changes
            if 'cpp' in file_types:
                print("[DevUpdate] C++ files changed, rebuilding...")
                success = self.build_system.build_cpp_components(changed_files)
                
                if success:
                    print("[DevUpdate] ✓ C++ rebuild successful")
                else:
                    print("[DevUpdate] ✗ C++ rebuild failed")
            
            # Handle Python changes
            if 'python' in file_types:
                print("[DevUpdate] Python files changed, hot reloading...")
                python_files = [f for f in changed_files if f.endswith('.py')]
                self.hot_reloader.reload_python_modules(python_files)
                print("[DevUpdate] ✓ Python modules reloaded")
            
            # Handle QML changes
            if 'qml' in file_types:
                print("[DevUpdate] QML files changed")
                # QML hot reload would be handled by Qt
            
            # Notify completion
            if self.on_update_complete:
                self.on_update_complete(changed_files, file_types)
            
            print("[DevUpdate] Update complete")
            
        except Exception as e:
            print(f"[DevUpdate] Update failed: {e}")
        
        finally:
            self.is_building = False


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'enabled': True,
        'project_root': '.',
        'watch_paths': ['./src', './include', './python'],
        'build_dir': 'build',
        'file_extensions': ['.cpp', '.hpp', '.py', '.qml'],
        'ignore_patterns': ['__pycache__', '.git', 'build', '.vscode'],
        'scan_interval': 2.0,
        'build_cooldown': 5.0,
        'cmake_args': []
    }
    
    # Initialize
    auto_update = DevAutoUpdate(config)
    
    # Callback for update completion
    def on_update(changed_files, file_types):
        print(f"Update completed for: {file_types}")
        print(f"Changed files: {changed_files}")
    
    # Start
    auto_update.start(on_update_complete=on_update)
    
    # Keep running
    try:
        print("Auto-update system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        auto_update.stop()
        print("Auto-update system stopped")
