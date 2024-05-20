import "./Navbar.css"
import {Link} from "react-router-dom"
const Navbar = () => {
  return ( 
    <header>
    <h1>Ask Your Senior</h1>
    <nav>
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/chatbot">Ask</Link></li>
        <li><Link to="/form">Help</Link></li>
      </ul>
    </nav>
  </header>
   );
}
 
export default Navbar;

