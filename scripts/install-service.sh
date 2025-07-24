#!/bin/bash

# GPU Stats Server - System Service Installation Script
# This script installs the GPU statistics server as a Linux systemd service

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="gpu-stats-server"
SERVICE_FILE="gpu-stats-server.service"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}GPU Stats Server Service Installer${NC}"
    echo -e "${PURPLE}================================${NC}"
    echo ""
}

# Check if running as root for installation
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root for safety reasons."
        print_status "It will use sudo when needed for system operations."
        exit 1
    fi
}

# Check if systemd is available
check_systemd() {
    if ! command -v systemctl &> /dev/null; then
        print_error "systemctl not found. This system doesn't appear to use systemd."
        exit 1
    fi
    print_success "systemd detected"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Python virtual environment exists
    if [ -d ".venv" ]; then
        print_success "Python virtual environment found"
    else
        print_error "Python virtual environment not found"
        print_status "Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        exit 1
    fi
    
    # Check if Python packages are installed
    if ./.venv/bin/python -c "import fastapi, uvicorn, dotenv" 2>/dev/null; then
        print_success "Required Python packages are installed"
    else
        print_error "Required Python packages missing"
        print_status "Installing packages..."
        ./.venv/bin/pip install -r requirements.txt
    fi
    
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        print_success "nvidia-smi found"
    else
        print_warning "nvidia-smi not found - GPU monitoring may not work"
    fi
    
    # Check if .env file exists
    if [ -f ".env" ]; then
        print_success ".env configuration file found"
    else
        print_warning ".env file not found, creating from example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            print_warning "No .env.example found, service will use defaults"
        fi
    fi
}

# Update service file with current paths
update_service_file() {
    print_status "Updating service file with current paths..."
    
    # Create a temporary service file with updated paths
    sed "s|/home/udg/projects/tt/gpu_stats|${CURRENT_DIR}|g" "${SERVICE_FILE}" > "${SERVICE_FILE}.tmp"
    
    # Update user if different
    CURRENT_USER=$(whoami)
    sed -i "s|User=udg|User=${CURRENT_USER}|g" "${SERVICE_FILE}.tmp"
    sed -i "s|Group=udg|Group=${CURRENT_USER}|g" "${SERVICE_FILE}.tmp"
    
    print_success "Service file updated for current environment"
}

# Install the service
install_service() {
    print_status "Installing systemd service..."
    
    # Copy service file to systemd directory
    sudo cp "${SERVICE_FILE}.tmp" "${SYSTEMD_DIR}/${SERVICE_FILE}"
    
    if [ $? -eq 0 ]; then
        print_success "Service file installed to ${SYSTEMD_DIR}/${SERVICE_FILE}"
    else
        print_error "Failed to install service file"
        exit 1
    fi
    
    # Set correct permissions
    sudo chmod 644 "${SYSTEMD_DIR}/${SERVICE_FILE}"
    
    # Reload systemd
    print_status "Reloading systemd daemon..."
    sudo systemctl daemon-reload
    
    if [ $? -eq 0 ]; then
        print_success "systemd daemon reloaded"
    else
        print_error "Failed to reload systemd daemon"
        exit 1
    fi
    
    # Clean up temporary file
    rm -f "${SERVICE_FILE}.tmp"
}

# Enable and start service
enable_service() {
    print_status "Enabling service to start on boot..."
    sudo systemctl enable "${SERVICE_NAME}"
    
    if [ $? -eq 0 ]; then
        print_success "Service enabled for automatic startup"
    else
        print_error "Failed to enable service"
        exit 1
    fi
}

# Start the service
start_service() {
    print_status "Starting ${SERVICE_NAME} service..."
    sudo systemctl start "${SERVICE_NAME}"
    
    if [ $? -eq 0 ]; then
        print_success "Service started successfully"
    else
        print_error "Failed to start service"
        print_status "Check service status with: sudo systemctl status ${SERVICE_NAME}"
        exit 1
    fi
}

# Check service status
check_service_status() {
    print_status "Checking service status..."
    sleep 2
    
    if sudo systemctl is-active --quiet "${SERVICE_NAME}"; then
        print_success "Service is running!"
        echo ""
        print_status "Service information:"
        sudo systemctl status "${SERVICE_NAME}" --no-pager -l
    else
        print_error "Service is not running"
        print_status "Check logs with: sudo journalctl -u ${SERVICE_NAME} -f"
        exit 1
    fi
}

# Show usage instructions
show_usage() {
    echo ""
    print_success "Installation completed successfully!"
    echo ""
    print_status "Service Management Commands:"
    echo "  Start service:    sudo systemctl start ${SERVICE_NAME}"
    echo "  Stop service:     sudo systemctl stop ${SERVICE_NAME}"
    echo "  Restart service:  sudo systemctl restart ${SERVICE_NAME}"
    echo "  Check status:     sudo systemctl status ${SERVICE_NAME}"
    echo "  View logs:        sudo journalctl -u ${SERVICE_NAME} -f"
    echo "  Disable service:  sudo systemctl disable ${SERVICE_NAME}"
    echo ""
    print_status "The GPU Statistics Server should now be accessible at:"
    
    # Read port from .env file if it exists
    if [ -f ".env" ]; then
        PORT=$(grep -E "^PORT=" .env | cut -d'=' -f2 | tr -d ' ')
        HOST=$(grep -E "^HOST=" .env | cut -d'=' -f2 | tr -d ' ')
        if [ -z "$PORT" ]; then PORT="5000"; fi
        if [ -z "$HOST" ] || [ "$HOST" = "0.0.0.0" ]; then HOST="localhost"; fi
    else
        PORT="5000"
        HOST="localhost"
    fi
    
    echo "  API:              http://${HOST}:${PORT}/"
    echo "  Documentation:    http://${HOST}:${PORT}/docs"
    echo "  Configuration:    http://${HOST}:${PORT}/config"
    echo ""
    print_status "To uninstall the service, run: sudo systemctl disable ${SERVICE_NAME} && sudo rm ${SYSTEMD_DIR}/${SERVICE_FILE}"
}

# Uninstall function
uninstall_service() {
    print_status "Uninstalling ${SERVICE_NAME} service..."
    
    # Stop the service
    sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null
    print_status "Service stopped"
    
    # Disable the service
    sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null
    print_status "Service disabled"
    
    # Remove service file
    sudo rm -f "${SYSTEMD_DIR}/${SERVICE_FILE}"
    print_success "Service file removed"
    
    # Reload systemd
    sudo systemctl daemon-reload
    print_success "systemd daemon reloaded"
    
    print_success "Service uninstalled successfully!"
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    case "${1:-install}" in
        "install")
            check_permissions
            check_systemd
            check_dependencies
            update_service_file
            install_service
            enable_service
            start_service
            check_service_status
            show_usage
            ;;
        "uninstall")
            print_status "Uninstalling GPU Stats Server service..."
            uninstall_service
            ;;
        "status")
            sudo systemctl status "${SERVICE_NAME}"
            ;;
        "logs")
            sudo journalctl -u "${SERVICE_NAME}" -f
            ;;
        *)
            echo "Usage: $0 {install|uninstall|status|logs}"
            echo ""
            echo "Commands:"
            echo "  install   - Install and start the service (default)"
            echo "  uninstall - Remove the service"
            echo "  status    - Show service status"
            echo "  logs      - Show service logs"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
