import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// TypeScript types for our database tables
export type Candidate = {
  id: string
  video_id: string
  platform: string
  title: string
  description: string | null
  thumbnail_url: string | null
  channel_name: string | null
  upload_date: string | null
  view_count: number | null
  like_count: number | null
  comment_count: number | null
  miles_score: number | null
  predicted_velocity: number | null
  tags: string[] | null
  duration_seconds: number | null
  scraped_at: string
  analyzed_at: string | null
  created_at: string
}

export type Query = {
  id: string
  user_id: string | null
  topic: string
  vibe: string | null
  additional_params: any
  recommendations: any
  matched_candidates: string[] | null
  created_at: string
}

export type Recommendation = {
  title: string
  reasoning: string
}

export type TopCandidate = {
  title: string
  thumbnail_url: string
  miles_score: number
  view_count: number
}

export type RecommendationResult = {
  success: boolean
  recommendations: Recommendation[]
  topCandidates: TopCandidate[]
}
