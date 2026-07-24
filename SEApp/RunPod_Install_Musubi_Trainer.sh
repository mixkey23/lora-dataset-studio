
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

cd /workspace

git clone --depth 1 https://github.com/FurkanGozukara/SECourses_Musubi_Trainer

cd SECourses_Musubi_Trainer

git reset --hard

git pull

git clone --depth 1 https://github.com/FurkanGozukara/convert_to_quant

cd convert_to_quant

git reset --hard

git pull

cd ..

git clone --depth 1 https://github.com/FurkanGozukara/musubi-tuner

cd musubi-tuner

git reset --hard

git pull

cd ..

# Reuse an existing stable Python 3.12 venv. Install/upgrade Python only when needed.
echo "Checking for an existing stable Python 3.12 virtual environment..."

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_READY=0
CAN_INSTALL_SYSTEM_PACKAGES=1
SUDO=()

if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO=(sudo)
    else
        CAN_INSTALL_SYSTEM_PACKAGES=0
    fi
fi

# PyTorch Inductor emits C++/OpenMP code on Linux. Install the host compiler and
# Ninja up front so torch.compile presets do not fail after model loading.
TOOLCHAIN_PACKAGES=()
if ! command -v g++ >/dev/null 2>&1; then
    TOOLCHAIN_PACKAGES+=(build-essential)
fi
if ! command -v ninja >/dev/null 2>&1; then
    TOOLCHAIN_PACKAGES+=(ninja-build)
fi
if [ "${#TOOLCHAIN_PACKAGES[@]}" -gt 0 ]; then
    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -ne 1 ]; then
        echo "Warning: torch.compile needs g++ with OpenMP and Ninja, but root access is unavailable."
    else
        echo "Installing the Linux torch.compile host toolchain..."
        if ! "${SUDO[@]}" apt-get update || ! "${SUDO[@]}" apt-get install -y "${TOOLCHAIN_PACKAGES[@]}"; then
            echo "Failed to install the Linux torch.compile host toolchain. Exiting..."
            exit 1
        fi
    fi
fi

# Remove the invalid filename created by older revisions. APT treats ".12" as
# an unsupported preferences-file extension and ignores it.
LEGACY_PYTHON_PREF="/etc/apt/preferences.d/deadsnakes-python3.12"
if [ -e "$LEGACY_PYTHON_PREF" ]; then
    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -eq 1 ]; then
        "${SUDO[@]}" rm -f "$LEGACY_PYTHON_PREF"
    else
        echo "Warning: cannot remove ignored APT preference file $LEGACY_PYTHON_PREF without root access."
    fi
fi

if [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
    echo "Reusing existing $("$VENV_PYTHON" --version 2>&1) virtual environment."
    VENV_READY=1
else
    if [ -x "$VENV_PYTHON" ]; then
        echo "Existing virtual environment is not using a stable Python 3.12 release."
    else
        echo "No existing Python 3.12 virtual environment was found."
    fi

    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -ne 1 ]; then
        echo "Root access or sudo is required to install Python 3.12. Exiting..."
        exit 1
    fi

    if command -v python3.12 >/dev/null 2>&1 && python3.12 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
        echo "Found $(python3.12 --version 2>&1); it will be upgraded if a newer stable package is available."
    elif command -v python3.12 >/dev/null 2>&1; then
        echo "Removing the existing prerelease Python 3.12 packages..."
        if ! "${SUDO[@]}" apt-get remove -y python3.12 python3.12-venv python3.12-dev; then
            echo "Failed to remove the prerelease Python 3.12 packages. Exiting..."
            exit 1
        fi
        if ! "${SUDO[@]}" apt-get autoremove -y; then
            echo "Failed to clean obsolete Python packages. Exiting..."
            exit 1
        fi
    fi

    echo "Adding deadsnakes PPA..."
    if ! "${SUDO[@]}" apt-get install -y software-properties-common; then
        echo "Failed to install software-properties-common. Exiting..."
        exit 1
    fi
    if ! "${SUDO[@]}" add-apt-repository -y ppa:deadsnakes/ppa; then
        echo "Failed to add the deadsnakes PPA. Exiting..."
        exit 1
    fi

    # Prefer stable Python 3.12 packages from deadsnakes over Ubuntu RC packages.
    "${SUDO[@]}" tee /etc/apt/preferences.d/deadsnakes-python312.pref >/dev/null <<'EOF'
Package: python3.12 python3.12-*
Pin: release o=LP-PPA-deadsnakes
Pin-Priority: 1000
EOF
    if [ "$?" -ne 0 ]; then
        echo "Failed to create the Python 3.12 APT preference. Exiting..."
        exit 1
    fi

    if ! "${SUDO[@]}" apt-get update; then
        echo "Failed to refresh APT package information. Exiting..."
        exit 1
    fi

    echo "Available Python 3.12 versions:"
    apt-cache policy python3.12

    echo "Installing the latest stable Python 3.12..."
    if ! "${SUDO[@]}" apt-get install -y python3.12 python3.12-venv python3.12-dev; then
        echo "Failed to install the Python 3.12 packages. Exiting..."
        exit 1
    fi

    if ! command -v python3.12 >/dev/null 2>&1 || ! python3.12 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
        echo "Failed to install a stable Python 3.12 interpreter. Exiting..."
        exit 1
    fi

    echo "Recreating the virtual environment with $(python3.12 --version 2>&1)..."
    if ! python3.12 -m venv --clear "$VENV_DIR"; then
        echo "Failed to recreate the Python 3.12 virtual environment. Exiting..."
        exit 1
    fi

    if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
        echo "Failed to create the Python 3.12 virtual environment. Exiting..."
        exit 1
    fi

    VENV_READY=1
fi

if [ "$VENV_READY" -ne 1 ]; then
    echo "Python 3.12 virtual environment setup failed. Exiting..."
    exit 1
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

python -m pip install uv

apt install python3.12-tk

cd ..

UV_CONCURRENT_INSTALLS=8 uv pip install -r requirements_musubi_trainer.txt --index-strategy unsafe-best-match

cd SECourses_Musubi_Trainer

cd musubi-tuner

uv pip install -e .

cd ..

cd convert_to_quant

uv pip install -e .

echo installation completed check for errors

unset LD_LIBRARY_PATH

cd ..

python gui.py --share
