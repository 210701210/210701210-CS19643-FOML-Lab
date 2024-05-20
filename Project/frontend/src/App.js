import {useState,useEffect} from "react";
import DataCollectionPage from "./pages/FormPage/DataCollectionPage";
import axios from "axios";
import Home from "./pages/HomePage/Home"
import ThankYou from "./pages/ThankYouPage/ThankYou"
import ChatBot from "./pages/ChatBotPage/ChatBot"
import {BrowserRouter,Routes,Route} from "react-router-dom"
const App = () => {
  // const [companyName,setCompanyName] = useState('');
  // const [rounds,setRounds] = useState('');
  // const [query,setQuery] = useState('');
  // const handleChange1 = (e)=>{
  //   setCompanyName(e.target.value);
  // }
  // const handleChange2 = (e)=>{
  //   setRounds(e.target.value);
  // }
  // const handleChange3 = (e)=>{
  //   setQuery(e.target.value);
  // }
  // const handleSubmit = async (e)=>{
  //   e.preventDefault();
  //   try {
  //     const response = await fetch("/api/append",{
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({ companyName,rounds }),
  //     })
  //     if (response.ok) {
  //       alert('CompanyName appended successfully!');
  //     } else {
  //       alert('Failed to append companyName.');
  //     }
  //   }
  //   catch(error)
  //   {
  //     console.error('Error:', error);
  //     alert('An error occurred. Please try again.');
  //   }
  // }
  // const handleQuery=async (e)=>{
  //   e.preventDefault();
  //   await axios.post("/api/getAns",{query})
  //   .then(res=>console.log(res))
  //   .catch(error=>console.log(error))
  // }
  return ( 
    <BrowserRouter>
      <Routes>
        <Route path="/form" element={<DataCollectionPage/>} />
        <Route path="/thankyou" element={<ThankYou/>} />
        <Route path="/chatbot" element={<ChatBot/>} />
        <Route path="/" element={<Home/>} />
      </Routes>
    </BrowserRouter>
   );
}
 
export default App;