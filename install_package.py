import streamlit as st
import subprocess
import os
import tempfile

st.title("Package Installer")

st.write("""
## Install .tar.gz Package
Upload a .tar.gz file to install it in the current environment.
""")

uploaded_file = st.file_uploader("Choose a .tar.gz file", type="tar.gz")

if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.write(f"File uploaded: {uploaded_file.name}")

    if st.button("Install Package"):
        with st.spinner("Installing package..."):
            try:
                # Run pip install on the tar.gz file
                result = subprocess.run(
                    ["pip", "install", tmp_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.success(
                    f"Package installed successfully!\n\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install package:\n\n{e.stderr}")
            finally:
                # Clean up the temporary file
                os.unlink(tmp_path)

st.write("""
### Alternative: Command Line Installation
You can also install a .tar.gz file using pip from the command line:
```
pip install /path/to/package.tar.gz
```
""")
