import "./ChatBot.css";
import {useState} from "react";
import axios from "axios";
const ChatBot = () => {
  const [query,setQuery] = useState('');
  const [response,setResponse] = useState(''); 
  const handleChange3 = (e)=>{
    setQuery(e.target.value);
  }
  const handleQuery=async (e)=>{
    e.preventDefault();
    await axios.post("/api/getAns",{query})
    .then(res=>setResponse(res.data.message))
    .catch(error=>console.log(error))
  }
  return ( 
    <>
      <form onSubmit={handleQuery} className="chatbot">
        <h1>ASK YOUR QUERIES</h1>
        <h2>QUERY:</h2>
        <input type="text" value={query} onChange={handleChange3} placeholder="Ask a query"/>
        <button type="submit" className="chatbot-btn">Submit</button>
        {response && 
        (
          <>
          <h2>RESPONSE:</h2>
          <div className="response">{response}</div>
          </>
        )
        }
      </form>
    </>
   );
}
 
export default ChatBot;