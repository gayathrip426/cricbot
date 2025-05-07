import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import AIChatbotUI from './ChatBot'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
        <AIChatbotUI /> 
    </>
  )
}

export default App
