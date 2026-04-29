import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { queryClient } from '@/api/client'
import { WardOverview } from '@/pages/WardOverview'
import { PatientDetail } from '@/pages/PatientDetail'
import { AlertCenter } from '@/pages/AlertCenter'
import { Admin } from '@/pages/Admin'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/ward/default" replace />} />
          <Route path="/ward/:wardId" element={<WardOverview />} />
          <Route path="/patient/:patientId" element={<PatientDetail />} />
          <Route path="/alerts" element={<AlertCenter />} />
          <Route path="/admin" element={<Admin />} />
          <Route
            path="/login"
            element={
              <div className="flex h-screen items-center justify-center text-gray-400">
                Login
              </div>
            }
          />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
