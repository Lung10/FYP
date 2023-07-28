import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, View, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system';

import Button from './components/Button';
import ImageViewer from './components/ImageViewer';

const PlaceholderImage = require('./assets/Picture1.png');

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');

  const pickImageAsync = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    } else {
      alert("You did not select any image.");
    }
  };

  const predictAgeAndGender = async () => {
    if (selectedImage) {
      try {
        const modelJSON = require('./assets/model/model.json'); // Update with your model path
        const modelWeights = require("./assets/model/group1-shard1of1.bin"); // Update with your model weights
        
        await tf.ready()

        const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights));

        const imageTensor = await loadImageTensor(selectedImage);
        
        const predictions = model.predict(imageTensor);
        const [genderProb, agePred] = await Promise.all([
          predictions[0].data(),
          predictions[1].data(),
        ]);
  
        const genderLabel = genderProb < 0.5 ? 'Male' : 'Female';

        console.log(Math.round(agePred[0]));
        console.log(genderLabel);
  
        setAge(Math.round(agePred[0]).toString());
        setGender(genderLabel);

      } catch (error) {
        console.error(error);
      }
    } else {
      alert("Please select an image first.");
    }
  };

  const loadImageTensor = async (fileUri) => {
    const imgB64 = await FileSystem.readAsStringAsync(fileUri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
    const raw = new Uint8Array(imgBuffer);
    const imgTensor = decodeJpeg(raw);
    const resizedTensor = tf.image.resizeBilinear(imgTensor, [128, 128]);
    const grayscaleTensor = tf.mean(resizedTensor, -1);
    const expandedTensor = grayscaleTensor.expandDims(0).expandDims(-1);
    const scaledTensor = expandedTensor.div(255.0);
    return scaledTensor;
  };
  
  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <ImageViewer
          placeholderImageSource={PlaceholderImage}
          selectedImage={selectedImage}
        />
        <View style={styles.infoContainer}>
          <Text style={styles.title}>Age and Gender prediction app</Text>
          <Text style={styles.age}>Age: {age}</Text>
          <Text style={styles.gender}>Gender: {gender}</Text>
        </View>
      </View>
      <View style={styles.footerContainer}>
        <Button theme="gallery" label="Choose a photo" onPress={pickImageAsync} />
        <Button theme="upload" label="Predict age and gender" onPress={predictAgeAndGender} />
      </View>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#25292e',
    alignItems: 'center',
  },
  imageContainer: {
    flex:1, 
    paddingTop: 58
  },
  infoContainer: {
    position: 'absolute',
    bottom: 10,
    left: 0,
    right: 0,
    backgroundColor: '#929292',
    borderRadius: 10,
    borderWidth: 2, 
    borderColor: '#F7F7F7',
    padding: 0,
  },
  footerContainer: {
    flex: 0.34,
    alignItems: 'center',
  },
  age: {
    color: '#000',
    fontSize: 16,
    height: 41,
    width: 270,
    textAlign: "left",
    marginTop: 8,
    marginLeft: 30
  },
  gender: {
    color: '#000',
    fontSize: 16,
    height: 41,
    width: 270,
    textAlign: "left",
    marginTop: 0,
    marginLeft: 30
  },
  title: {
    color: '#000',
    fontSize: 18,
    fontWeight: 'bold',
    height: 41,
    width: 270,
    textAlign: "center",
    marginTop: 10,
    marginLeft: 15
  },
});
