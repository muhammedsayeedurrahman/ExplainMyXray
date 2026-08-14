import { supabase } from './client';
import { Role } from '@/config/roles';

export interface SignUpData {
  email: string;
  password: string;
  fullName: string;
  role: Role;
  workspaceName?: string; // For first user creating workspace
}

export interface SignInData {
  email: string;
  password: string;
}

/**
 * Sign up new user and create workspace if first user
 */
export async function signUp(data: SignUpData) {
  const { email, password, fullName, role, workspaceName } = data;

  // Sign up with Supabase Auth
  const { data: authData, error: authError } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        full_name: fullName,
        role: role
      }
    }
  });

  if (authError) {
    throw new Error(authError.message);
  }

  if (!authData.user) {
    throw new Error('Sign up failed');
  }

  // If workspace name provided, create workspace (first user scenario)
  if (workspaceName) {
    const { data: workspace, error: workspaceError } = await supabase
      .from('workspaces')
      .insert([{ name: workspaceName }])
      .select()
      .single();

    if (workspaceError) {
      throw new Error(workspaceError.message);
    }

    // Update user with workspace_id
    const { error: updateError } = await supabase
      .from('users')
      .update({ workspace_id: workspace.id })
      .eq('id', authData.user.id);

    if (updateError) {
      throw new Error(updateError.message);
    }
  }

  return authData;
}

/**
 * Sign in existing user
 */
export async function signIn(data: SignInData) {
  const { email, password } = data;

  const { data: authData, error } = await supabase.auth.signInWithPassword({
    email,
    password
  });

  if (error) {
    throw new Error(error.message);
  }

  return authData;
}

/**
 * Sign out current user
 */
export async function signOut() {
  const { error } = await supabase.auth.signOut();

  if (error) {
    throw new Error(error.message);
  }
}

/**
 * Get current session
 */
export async function getSession() {
  const { data: { session }, error } = await supabase.auth.getSession();

  if (error) {
    throw new Error(error.message);
  }

  return session;
}

/**
 * Get current user with profile data
 */
export async function getCurrentUser() {
  const session = await getSession();

  if (!session?.user) {
    return null;
  }

  // Get user profile from users table
  const { data: profile, error } = await supabase
    .from('users')
    .select('*')
    .eq('id', session.user.id)
    .single();

  if (error) {
    throw new Error(error.message);
  }

  return {
    ...session.user,
    profile
  };
}

/**
 * Update user profile
 */
export async function updateProfile(updates: {
  full_name?: string;
  avatar_url?: string;
  phone?: string;
}) {
  const session = await getSession();

  if (!session?.user) {
    throw new Error('Not authenticated');
  }

  const { error } = await supabase
    .from('users')
    .update(updates)
    .eq('id', session.user.id);

  if (error) {
    throw new Error(error.message);
  }
}

/**
 * Change password
 */
export async function changePassword(newPassword: string) {
  const { error } = await supabase.auth.updateUser({
    password: newPassword
  });

  if (error) {
    throw new Error(error.message);
  }
}
