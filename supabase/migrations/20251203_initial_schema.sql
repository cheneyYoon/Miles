-- Users table (extends Supabase Auth)
CREATE TABLE public.users (
  id UUID REFERENCES auth.users(id) PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  display_name TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Video candidates table (scraped data)
CREATE TABLE public.candidates (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  video_id TEXT UNIQUE NOT NULL,           -- YouTube video ID
  platform TEXT DEFAULT 'youtube',          -- 'youtube' or 'tiktok'
  title TEXT NOT NULL,
  description TEXT,
  thumbnail_url TEXT,
  channel_name TEXT,
  upload_date TIMESTAMPTZ,

  -- Engagement metrics (from scraper)
  view_count BIGINT,
  like_count BIGINT,
  comment_count BIGINT,

  -- Miles predictions
  miles_score FLOAT,                        -- Virality score [0,1]
  predicted_velocity FLOAT,                 -- Predicted view velocity

  -- Metadata
  tags TEXT[],
  duration_seconds INT,
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  analyzed_at TIMESTAMPTZ,                  -- When ML inference ran

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User queries table (track user requests)
CREATE TABLE public.queries (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES public.users(id),

  -- User input
  topic TEXT NOT NULL,                      -- e.g., "tech reviews"
  vibe TEXT,                                -- e.g., "funny", "educational"
  additional_params JSONB,                  -- Flexible storage

  -- System response
  recommendations JSONB,                    -- LLM-generated suggestions
  matched_candidates UUID[],                -- Array of candidate IDs

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recommendation cache table (for performance optimization)
CREATE TABLE public.recommendation_cache (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  cache_key TEXT UNIQUE NOT NULL,
  result JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analytics table (for tracking usage)
CREATE TABLE public.analytics (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  event_type TEXT NOT NULL,
  user_id UUID REFERENCES public.users(id),
  topic TEXT,
  processing_time_ms INT,
  success BOOLEAN DEFAULT TRUE,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_candidates_miles_score ON public.candidates(miles_score DESC);
CREATE INDEX idx_candidates_platform_score ON public.candidates(platform, miles_score DESC);
CREATE INDEX idx_candidates_scraped_at ON public.candidates(scraped_at DESC);
CREATE INDEX idx_candidates_analyzed_at ON public.candidates(analyzed_at) WHERE analyzed_at IS NULL;
CREATE INDEX idx_queries_user_id ON public.queries(user_id);
CREATE INDEX idx_queries_created_at ON public.queries(created_at DESC);
CREATE INDEX idx_recommendation_cache_key ON public.recommendation_cache(cache_key);
CREATE INDEX idx_recommendation_cache_created_at ON public.recommendation_cache(created_at DESC);
CREATE INDEX idx_analytics_event_type ON public.analytics(event_type, created_at DESC);

-- Row Level Security (RLS) Policies
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.recommendation_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analytics ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can read own data" ON public.users
  FOR SELECT USING (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile" ON public.users
  FOR UPDATE USING (auth.uid() = id);

-- Anyone can read candidates (public data)
CREATE POLICY "Candidates are publicly readable" ON public.candidates
  FOR SELECT USING (true);

-- Only service role can insert/update candidates
CREATE POLICY "Service role can manage candidates" ON public.candidates
  FOR ALL USING (auth.role() = 'service_role');

-- Users can read their own queries
CREATE POLICY "Users can read own queries" ON public.queries
  FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

-- Users can insert their own queries
CREATE POLICY "Users can insert own queries" ON public.queries
  FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Cache is readable by anyone (for performance)
CREATE POLICY "Cache is publicly readable" ON public.recommendation_cache
  FOR SELECT USING (true);

-- Only service role can manage cache
CREATE POLICY "Service role can manage cache" ON public.recommendation_cache
  FOR ALL USING (auth.role() = 'service_role');

-- Analytics is write-only for users
CREATE POLICY "Anyone can write analytics" ON public.analytics
  FOR INSERT WITH CHECK (true);

-- Only service role can read analytics
CREATE POLICY "Service role can read analytics" ON public.analytics
  FOR SELECT USING (auth.role() = 'service_role');

-- Function to clean old cache entries (optional, for maintenance)
CREATE OR REPLACE FUNCTION clean_old_cache()
RETURNS void AS $$
BEGIN
  DELETE FROM public.recommendation_cache
  WHERE created_at < NOW() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON public.users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
