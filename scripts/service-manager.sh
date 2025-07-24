#!/bin/bash

# GPU Stats Server - Service Management Script
# Quick commands to manage the GPU statistics server service

SERVICE_NAME="gpu-stats-server"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    echo "GPU Stats Server Service Manager"
    echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the service"
    echo "  stop     - Stop the service"
    echo "  restart  - Restart the service"
    echo "  status   - Show service status"
    echo "  logs     - Show service logs (follow mode)"
    echo "  enable   - Enable service to start on boot"
    echo "  disable  - Disable service from starting on boot"
    echo ""
}

case "${1}" in
    "start")
        print_status "Starting ${SERVICE_NAME} service..."
        sudo systemctl start "${SERVICE_NAME}"
        if [ $? -eq 0 ]; then
            print_success "Service started"
            sudo systemctl status "${SERVICE_NAME}" --no-pager -l
        else
            print_error "Failed to start service"
        fi
        ;;
    "stop")
        print_status "Stopping ${SERVICE_NAME} service..."
        sudo systemctl stop "${SERVICE_NAME}"
        if [ $? -eq 0 ]; then
            print_success "Service stopped"
        else
            print_error "Failed to stop service"
        fi
        ;;
    "restart")
        print_status "Restarting ${SERVICE_NAME} service..."
        sudo systemctl restart "${SERVICE_NAME}"
        if [ $? -eq 0 ]; then
            print_success "Service restarted"
            sudo systemctl status "${SERVICE_NAME}" --no-pager -l
        else
            print_error "Failed to restart service"
        fi
        ;;
    "status")
        sudo systemctl status "${SERVICE_NAME}" --no-pager -l
        ;;
    "logs")
        print_status "Showing logs for ${SERVICE_NAME} service (Ctrl+C to exit)..."
        sudo journalctl -u "${SERVICE_NAME}" -f
        ;;
    "enable")
        print_status "Enabling ${SERVICE_NAME} service for automatic startup..."
        sudo systemctl enable "${SERVICE_NAME}"
        if [ $? -eq 0 ]; then
            print_success "Service enabled"
        else
            print_error "Failed to enable service"
        fi
        ;;
    "disable")
        print_status "Disabling ${SERVICE_NAME} service from automatic startup..."
        sudo systemctl disable "${SERVICE_NAME}"
        if [ $? -eq 0 ]; then
            print_success "Service disabled"
        else
            print_error "Failed to disable service"
        fi
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
