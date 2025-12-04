'use client'

import { useState } from 'react'
import { supabase, type Recommendation, type TopCandidate } from '@/lib/supabase'

export default function Home() {
  const [topic, setTopic] = useState('')
  const [vibe, setVibe] = useState('')
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [topCandidates, setTopCandidates] = useState<TopCandidate[]>([])
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    // TODO: Call Supabase Edge Function for recommendations (Phase 4)
    // For now, return mock data
    setTimeout(() => {
      setRecommendations([
        {
          title: '5 Tech Trends That Will Change Everything in 2025',
          reasoning: 'Based on analysis of 50+ viral tech videos, short countdown formats with bold predictions perform exceptionally well. Your timing aligns with year-end content trends.'
        },
        {
          title: 'I Tested Every AI Tool So You Don\'t Have To',
          reasoning: 'Comparison videos with first-person testing show 3x higher engagement. The "so you don\'t have to" formula creates immediate value proposition in the title.'
        },
        {
          title: 'This Hidden iPhone Feature is a Game Changer',
          reasoning: 'Mystery-driven titles with product-specific tips average 85% viral score. Appeals to curiosity while targeting a massive audience (iPhone users).'
        }
      ])
      setTopCandidates([
        {
          title: 'iPhone 15 Pro - 10 Features Apple Didn\'t Tell You',
          thumbnail_url: 'https://images.unsplash.com/photo-1592286927505-b0c2d374eb84?w=400&h=300&fit=crop',
          miles_score: 0.89,
          view_count: 2400000
        },
        {
          title: 'ChatGPT Just Changed Everything (Again)',
          thumbnail_url: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400&h=300&fit=crop',
          miles_score: 0.87,
          view_count: 1800000
        },
        {
          title: 'Why Everyone is Switching to This App',
          thumbnail_url: 'https://images.unsplash.com/photo-1551650975-87deedd944c3?w=400&h=300&fit=crop',
          miles_score: 0.85,
          view_count: 1500000
        }
      ])
      setLoading(false)
    }, 1500)
  }

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto bg-gradient-to-br from-blue-50 to-purple-50">
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Miles AI
        </h1>
        <p className="text-gray-600 text-lg">
          AI-Powered Viral Video Recommendations
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Analyze trending content and get personalized video ideas
        </p>
      </header>

      <form onSubmit={handleSubmit} className="max-w-2xl mx-auto space-y-6 mb-12 bg-white p-8 rounded-2xl shadow-lg">
        <div>
          <label className="block text-sm font-semibold mb-3 text-gray-700">
            What&apos;s your content topic?
          </label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., tech reviews, cooking, fitness, travel"
            className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:outline-none transition"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-semibold mb-3 text-gray-700">
            Choose your vibe
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {['funny', 'educational', 'inspirational', 'dramatic'].map(v => (
              <button
                key={v}
                type="button"
                onClick={() => setVibe(v === vibe ? '' : v)}
                className={`p-3 rounded-xl border-2 transition capitalize ${
                  vibe === v
                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Analyzing trends...
            </span>
          ) : (
            'Get Viral Ideas'
          )}
        </button>

        {error && (
          <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl text-red-700">
            {error}
          </div>
        )}
      </form>

      {recommendations.length > 0 && (
        <div className="max-w-4xl mx-auto space-y-8">
          <section>
            <h2 className="text-3xl font-bold mb-6 text-gray-800">
              Your Viral Video Ideas
            </h2>
            <div className="space-y-4">
              {recommendations.map((rec, idx) => (
                <div key={idx} className="p-6 bg-white rounded-2xl shadow-md hover:shadow-xl transition">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center text-white font-bold text-xl">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-bold text-xl mb-2 text-gray-800">
                        {rec.title}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">
                        {rec.reasoning}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {topCandidates.length > 0 && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-gray-800">
                Inspired by these trending videos
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                {topCandidates.map((video, idx) => (
                  <div key={idx} className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition">
                    <img
                      src={video.thumbnail_url}
                      alt={video.title}
                      className="w-full h-48 object-cover"
                    />
                    <div className="p-4">
                      <p className="font-semibold text-sm mb-2 line-clamp-2">
                        {video.title}
                      </p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{(video.view_count / 1000).toFixed(0)}K views</span>
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full font-semibold">
                          {(video.miles_score * 100).toFixed(0)}% viral
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </main>
  )
}
