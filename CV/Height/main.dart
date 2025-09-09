import 'dart:io';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:path_provider/path_provider.dart';
import 'package:csv/csv.dart';

class HeightDetector extends StatefulWidget {
  const HeightDetector({Key? key}) : super(key: key);

  @override
  _HeightDetectorState createState() => _HeightDetectorState();
}

class _HeightDetectorState extends State<HeightDetector> {
  late CameraController _cameraController;
  late Future<void> _initializeControllerFuture;

  final poseDetector = PoseDetector(options: PoseDetectorOptions());
  final FlutterTts tts = FlutterTts();

  double pixelToCm = 0.5; // calibration factor
  int? stableHeight;
  DateTime? stableStart;
  final int stabilityTime = 5; // seconds

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
    _initializeControllerFuture = _cameraController.initialize();
    await _initializeControllerFuture;
    _cameraController.startImageStream(_processCameraImage);
  }

  Future<void> _processCameraImage(CameraImage image) async {
    final inputImage = _convertCameraImage(image);

    final poses = await poseDetector.processImage(inputImage);

    for (Pose pose in poses) {
      final nose = pose.landmarks[PoseLandmarkType.nose];
      final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
      final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];

      if (nose != null && (leftAnkle != null || rightAnkle != null)) {
        final bottom = leftAnkle ?? rightAnkle;
        final distance = sqrt(
          pow(nose.x - bottom.x, 2) + pow(nose.y - bottom.y, 2),
        );
        final heightCm = (distance * pixelToCm).round();

        _checkStability(heightCm);
      }
    }
  }

  void _checkStability(int heightCm) async {
    final now = DateTime.now();
    if (stableHeight == null || (heightCm - stableHeight!).abs() > 2) {
      stableHeight = heightCm;
      stableStart = now;
    } else {
      if (stableStart != null &&
          now.difference(stableStart!).inSeconds >= stabilityTime) {
        await tts.speak("Your final height is $stableHeight centimeters");
        await _saveHeight(stableHeight!);
        _cameraController.stopImageStream();
      }
    }
  }

  Future<void> _saveHeight(int height) async {
    final directory = await getApplicationDocumentsDirectory();
    final logFile = File('${directory.path}/height_log.csv');

    if (!await logFile.exists()) {
      await logFile.writeAsString("Timestamp,Height (cm)\n");
    }

    final timestamp = DateTime.now().toIso8601String();
    final row = "$timestamp,$height\n";
    await logFile.writeAsString(row, mode: FileMode.append);
  }

  InputImage _convertCameraImage(CameraImage image) {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    final Size imageSize = Size(image.width.toDouble(), image.height.toDouble());
    final camera = _cameraController.description;
    final imageRotation =
        InputImageRotationValue.fromRawValue(camera.sensorOrientation) ??
            InputImageRotation.rotation0deg;

    final inputImageFormat =
        InputImageFormatValue.fromRawValue(image.format.raw) ??
            InputImageFormat.nv21;

    final planeData = image.planes.map(
      (Plane plane) {
        return InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        );
      },
    ).toList();

    final inputImageData = InputImageData(
      size: imageSize,
      imageRotation: imageRotation,
      inputImageFormat: inputImageFormat,
      planeData: planeData,
    );

    return InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);
  }

  @override
  void dispose() {
    _cameraController.dispose();
    poseDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return CameraPreview(_cameraController);
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}
