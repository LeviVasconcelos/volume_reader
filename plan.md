# Surfboard Volume Calculator

## Overview
A mobile tool for computing surfboard volumes using photogrammetry-based 3D reconstruction.

## Requirements
- **Accuracy target:** ±0.5 liters
- **Platform:** Mobile app (iOS/Android)
- **Processing:** Cloud-based

## Architecture

### Mobile App (React Native)
- Camera capture with real-time AR overlay guidance
- Shows coverage indicators, angle suggestions, quality feedback
- Detects physical reference marker for scale calibration
- Uploads images to cloud backend
- Displays computed volume results

### Backend (Python + FastAPI)
- Receives captured images
- 3D reconstruction pipeline:
  1. Feature extraction (SIFT/ORB)
  2. Feature matching across images
  3. Structure from Motion (SfM)
  4. Dense reconstruction
  5. Mesh generation
  6. Scale calibration using reference marker
  7. Volume calculation (mesh integration)
- Returns volume in liters

## Capture Method
**Phase 1:** Multi-view photogrammetry
- User walks around surfboard taking 20-40 photos
- AR guidance ensures good coverage and angles
- Physical reference marker (credit card size) placed next to board for scale

**Phase 2 (future):** SLAM-based video capture
- More natural capture flow
- Real-time reconstruction feedback

## Scale Calibration
Two methods supported (user chooses):

**Option A: Physical reference marker**
- Known-size marker (credit card, printed pattern) placed next to board
- Auto-detected in images to establish scale
- Most reliable for ±0.5L accuracy

**Option B: User-provided dimensions**
- User enters known board dimensions (length, width, thickness)
- System scales the 3D model to match
- Useful when marker unavailable or user has exact specs
- Cross-validates dimensions if multiple provided

## Tech Stack
- **Mobile:** React Native with camera/AR libraries
- **Backend:** Python, FastAPI
- **3D Processing:** OpenCV, Open3D, COLMAP or similar SfM library
- **Deployment:** Cloud (AWS/GCP) with GPU instances for reconstruction

## Validation
- Multiple surfboards with known volumes available for testing
- Will create test dataset and accuracy benchmarks

## Development Phases
1. Backend reconstruction pipeline (proof of concept)
2. Volume calculation from mesh
3. Scale calibration with reference marker
4. Mobile app with basic capture
5. AR guidance overlay
6. Integration and testing
7. Accuracy optimization
