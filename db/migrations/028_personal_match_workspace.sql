-- Personal notes and watchlist for the authenticated single user.
begin;

create table if not exists public.personal_match_workspace (
    user_id uuid not null references auth.users(id) on delete cascade,
    match_id bigint not null references public.matches(id) on delete cascade,
    note text not null default '',
    is_watched boolean not null default false,
    updated_at timestamptz not null default now(),
    primary key (user_id, match_id)
);

alter table public.personal_match_workspace enable row level security;

drop policy if exists personal_match_workspace_select on public.personal_match_workspace;
create policy personal_match_workspace_select
    on public.personal_match_workspace for select
    using (auth.uid() = user_id);

drop policy if exists personal_match_workspace_insert on public.personal_match_workspace;
create policy personal_match_workspace_insert
    on public.personal_match_workspace for insert
    with check (auth.uid() = user_id);

drop policy if exists personal_match_workspace_update on public.personal_match_workspace;
create policy personal_match_workspace_update
    on public.personal_match_workspace for update
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists personal_match_workspace_delete on public.personal_match_workspace;
create policy personal_match_workspace_delete
    on public.personal_match_workspace for delete
    using (auth.uid() = user_id);

create index if not exists personal_match_workspace_user_updated_idx
    on public.personal_match_workspace (user_id, updated_at desc);

commit;
