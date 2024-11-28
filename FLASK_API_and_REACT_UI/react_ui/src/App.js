import './App.css';
import PlantUi from './pages/plant_ui';
import PlantUi_Without_Click_Button from './pages/plant_ui_without_click_button';
import PlantUiPytorch from './pages/PlantUiPytorch';
import { BrowserRouter, Route, Routes } from 'react-router-dom'; // Import BrowserRouter and Route

function App() {
  return (
    <BrowserRouter>
      <>
        {/* Define routes for your components */}
        <Routes>
          <Route path="/" element={<PlantUiPytorch />} />
          <Route path="/plant_classification_Pytorch_project_ui" element={<PlantUiPytorch />} />
          {/* You can add more routes as needed */}
        </Routes>
      </>
    </BrowserRouter>
  );
}

export default App;
